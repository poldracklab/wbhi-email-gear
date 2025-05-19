#!/usr/bin/env python

import flywheel_gear_toolkit
import flywheel
import logging
import pip
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime, date, timedelta
import pandas as pd
from redcap import Project
from flywheel import FileListOutput, ProjectOutput

pip.main(["install", "--upgrade", "git+https://github.com/poldracklab/wbhi-utils.git"])
from wbhiutils import parse_dicom_hdr  # noqa: E402
from wbhiutils.constants import (  # noqa: E402
    EMAIL_DICT,
    ADMIN_EMAIL,
    REDCAP_API_URL,
    REDCAP_KEY,
    SITE_KEY_REVERSE,
    SITE_LIST,
    DATE_FORMAT_FW,
    DATE_FORMAT_RC,
)

log = logging.getLogger(__name__)

DATAVIEW_COLUMNS = (
    "subject.label",
    "session.id",
    "session.tags",
    "file.file_id",
    "file.tags",
    "file.type",
    "file.created",
    "acquisition.label",
)

DICOM_FUNCTION_DICT = {
    "date": lambda dcm_hdr, site: datetime.strptime(
        dcm_hdr["StudyDate"], DATE_FORMAT_FW
    ),
    "am_pm": lambda dcm_hdr, site: "am"
    if float(dcm_hdr["StudyTime"]) < 120000
    else "pm",
    "pi_id": lambda dcm_hdr, site: parse_dicom_hdr.parse_pi(dcm_hdr, site),
    "sub_id": lambda dcm_hdr, site: parse_dicom_hdr.parse_sub(dcm_hdr, site),
}


def create_view_df(
    container, columns: tuple, filter=None, container_type="acquisition"
) -> pd.DataFrame:
    """Get unique labels for all acquisitions in the container.

    This is done using a single Data View which is more efficient than iterating through
    all acquisitions, sessions, and subjects. This prevents time-out errors in large projects.
    """

    builder = flywheel.ViewBuilder(
        container=container_type,
        filename="*.*",
        match="all",
        filter=filter,
        process_files=False,
        include_ids=False,
        include_labels=False,
    )
    for c in columns:
        builder.column(src=c)

    view = builder.build()
    return client.read_view_dataframe(view, container.id)


def create_first_dcm_df(dcm_df: pd.DataFrame) -> pd.DataFrame:
    """Return a df containing the first file from each session in dcm_df."""
    first_df = dcm_df.copy()
    first_df = first_df.sort_values(by=["session.id", "file.created"])
    return first_df.drop_duplicates(subset="session.id")


def get_acq_or_file_path(container) -> str:
    """Take a container and return its path."""
    project_label = client.get_project(container.parents.project).label
    sub_label = client.get_subject(container.parents.subject).label
    ses_label = client.get_session(container.parents.session).label

    if container.container_type == "acq":
        return f"{project_label}/{sub_label}/{ses_label}/{container.label}"
    elif container.container_type == "file":
        acq_label = client.get_acquisition(container.parents.acquisition).label
        return f"{project_label}/{sub_label}/{ses_label}/{acq_label}/{container.name}/"


def get_last_job_date() -> str:
    """Returns the date of the most recent successful run of this gear."""
    date_format = "%Y-%m-%d"
    cutoff_date = date.today() - timedelta(days=30)
    cutoff_date_str = cutoff_date.strftime(date_format)
    recent_jobs = client.jobs.find(
        f"created>{cutoff_date_str},gear_info.name=wbhi-email,state=complete,config.config.test_run!=true"
    )
    if recent_jobs:
        last_job_date = sorted([j.created for j in recent_jobs])[-1]
        return last_job_date.strftime(date_format)
    else:
        # If no jobs found with 30 days, return date of one week ago
        one_week_ago = date.today() - timedelta(days=7)
        return one_week_ago.strftime(date_format)


def get_hdr_fields(dicom: FileListOutput, site: str) -> dict:
    """Get relevant fields from dicom header of an acquisition."""
    # Reload the dicom file to ensure dicom header is loaded
    dicom = dicom.reload()
    dcm = get_acq_or_file_path(dicom)

    if "file-classifier" not in dicom.tags or "header" not in dicom.info:
        log.error("File-classifier gear has not been run on %s", dcm)
        return {"error": "FILE_CLASSIFIER_NOT_RUN"}

    dcm_hdr = dicom.info["header"]["dicom"]
    meta = {"error": None, "site": site, "ses_id": dicom.parents.session}

    try:
        meta["pi_id"] = parse_dicom_hdr.parse_pi(dcm_hdr, site).casefold()
    except (KeyError, ValueError):
        log.debug("%s problem fetching PI ID", dcm)
        meta["error"] = "MISSING_DICOM_FIELDS"

    try:
        meta["sub_id"] = parse_dicom_hdr.parse_sub(dcm_hdr, site).casefold()
    except KeyError:
        log.debug("%s problem fetching SUB ID", dcm)
        meta["error"] = "MISSING_DICOM_FIELDS"

    try:
        meta["date"] = datetime.strptime(dcm_hdr["StudyDate"], DATE_FORMAT_FW)
    except KeyError:
        log.debug("%s problem fetching DATE", dcm)
        meta["error"] = "MISSING_DICOM_FIELDS"

    try:
        meta["am_pm"] = "am" if float(dcm_hdr["StudyTime"]) < 120000 else "pm"
    except KeyError:
        log.debug("%s problem fetching AM/PM", dcm)
        meta["error"] = "MISSING_DICOM_FIELDS"

    return meta


def create_new_matches_df(pre_deid_project: ProjectOutput) -> pd.DataFrame:
    """Return a df containing new matches since the last email was sent."""
    filter = "session.tags!=email,file.type=dicom"
    dcm_df = create_view_df(pre_deid_project, DATAVIEW_COLUMNS, filter)
    if dcm_df.empty:
        return dcm_df
    first_dcm_df = create_first_dcm_df(dcm_df)

    hdr_list = []
    for index, row in first_dcm_df.iterrows():
        dicom = client.get_file(row["file.file_id"])
        site = SITE_KEY_REVERSE[row["subject.label"][0]]
        hdr_fields = get_hdr_fields(dicom, site)
        hdr_list.append(hdr_fields)

    hdr_df = pd.DataFrame(hdr_list)
    hdr_df = hdr_df.drop("error", axis=1)

    return hdr_df.sort_values("date")


def create_just_fw_df() -> pd.DataFrame:
    """Return a df containing unmatched flywheel sessions."""
    today = datetime.today()
    hdr_list = []
    for site in SITE_LIST:
        project = client.lookup(f"{site}/Inbound Data")
        filter = "file.type=dicom"
        dcm_df = create_view_df(project, DATAVIEW_COLUMNS, filter)
        first_file_df = create_first_dcm_df(dcm_df)
        if first_file_df.empty:
            continue

        for file_id in first_file_df["file.file_id"]:
            file = client.get_file(file_id)
            hdr_fields = get_hdr_fields(file, site)
            if hdr_fields["error"] == "FILE_CLASSIFIER_NOT_RUN":
                continue
            if "date" in hdr_fields:
                delta = today - hdr_fields["date"]
                if delta >= timedelta(days=2):
                    hdr_list.append(hdr_fields)
            else:
                hdr_list.append(hdr_fields)

    hdr_df = pd.DataFrame(hdr_list)
    hdr_df = hdr_df.drop("error", axis=1)

    return hdr_df.sort_values("date")


def create_just_rc_df(redcap_project: Project) -> pd.DataFrame:
    """Return a df containing unmatched redcap records."""
    # Since there's no way to reset a field to '', occassionally rid will be ' '
    # if it's value was deleted. Thus, we need to check for both cases.
    filter_logic = "[rid] = '' or [rid] = ' '"
    redcap_data = redcap_project.export_records(filter_logic=filter_logic)
    just_rc_list = []

    for record in redcap_data:
        redcap_id = record.get("participant_id")
        site = record.get("site")

        if site:
            mri_pi_field = f"mri_pi_{site}"
            # Some labels may be empty strings
            if record[mri_pi_field] != "99":
                pi_id = record[mri_pi_field].casefold()
            else:
                pi_id = record[f"{mri_pi_field}_other"].casefold()
        else:
            pi_id = None

        record_dict = {
            "site": site,
            "redcap_id": redcap_id,
            "pi_id": pi_id,
            "sub_id": record.get("mri").casefold(),
        }

        # Dates may be blank
        try:
            record_dict["date"] = datetime.strptime(record["mri_date"], DATE_FORMAT_RC)
        except ValueError:
            log.error(
                "Could not extract date information for record number %s", redcap_id
            )
            record_dict["date"] = pd.NaT

        try:
            record_dict["am_pm"] = REDCAP_KEY["am_pm"][record["mri_ampm"]]
        except ValueError:
            log.error(
                "Could not extract am/pm information for record number %s", redcap_id
            )
            record_dict["am_pm"] = pd.NaT

        just_rc_list.append(record_dict)

    return pd.DataFrame(just_rc_list).sort_values("date")


def create_failed_jobs_df() -> pd.DataFrame():
    """Return a df containing failed gear runs since the last email was sent."""
    last_email_job_date = get_last_job_date()
    failed_jobs = client.jobs.find(f'created>{last_email_job_date},state=failed')

    failed_jobs_dict_list = []
    for job in failed_jobs:
        subject = None
        session = None
        if 'subject' in job.parents:
            subject = client.get_subject(job.parents.subject).label
        if 'session' in job.parents:
            session = client.get_session(job.parents.session).label

        job_dict = {
            'name': job.gear_info.name,
            'id': job.id,
            'group': job.parents.group,
            'project': client.get_project(job.parents.project).label,
            'subject': subject,
            'session': session,
            'date': job.created.date()
        }

        failed_jobs_dict_list.append(job_dict)

    return pd.DataFrame(failed_jobs_dict_list).sort_values('date')


def create_long_interval_df(pre_deid_project: ProjectOutput) -> pd.DataFrame():
    """Return a df of all sessions in wbhi/pre-deid containing the tag 'long-redcap-interval_unsent'."""
    long_interval_df = create_view_df(
        pre_deid_project,
        ["session.id"],
        filter="session.tags=long-redcap-interval_unsent",
        container_type="session",
    )
    long_interval_df = long_interval_df.rename(columns={"session.id":"ses_id"})
    return long_interval_df.drop("errors", axis=1)


def send_email(subject, html_content, sender, recipients, password, files=None) -> None:
    """Send an email containing html content."""
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html_content, "html"))

    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(fil.read(), Name=os.path.basename(f))
        part["Content-Disposition"] = 'attachment; filename="%s"' % os.path.basename(f)
        msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp_server:
        smtp_server.login(sender, password)
        smtp_server.sendmail(sender, recipients, msg.as_string())

    log.info(f"Email sent to {recipients}")


def send_wbhi_email(
    new_matches_df: pd.DataFrame,
    just_rc_df: pd.DataFrame,
    just_fw_df: pd.DataFrame,
    site: str,
    test_run=False,
) -> None:
    """Send wbhi email updated to sites and/or admins."""
    new_matches_df_copy = new_matches_df.copy()
    just_rc_df_copy = just_rc_df.copy()
    just_fw_df_copy = just_fw_df.copy()

    if site != "admin":
        if not new_matches_df_copy.empty:
            new_matches_df_copy = new_matches_df_copy[
                new_matches_df_copy["site"] == site
            ]
        if not just_rc_df_copy.empty:
            just_rc_df_copy = just_rc_df_copy[just_rc_df_copy["site"] == site]
        if not just_fw_df_copy.empty:
            just_fw_df_copy = just_fw_df_copy[just_fw_df_copy["site"] == site]

    to_kwargs = {"index": False, "na_rep": ""}
    new_matches_html = new_matches_df_copy.to_html(**to_kwargs)
    just_rc_html = just_rc_df_copy.to_html(**to_kwargs)
    just_fw_html = just_fw_df_copy.to_html(**to_kwargs)

    csv_path = os.path.join(os.environ["FLYWHEEL"], "csv")
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    new_matches_df_copy.to_csv(os.path.join(csv_path, "matches.csv"), **to_kwargs)
    just_rc_df_copy.to_csv(os.path.join(csv_path, "redcap_unmatched.csv"), **to_kwargs)
    just_fw_df_copy.to_csv(
        os.path.join(csv_path, "flywheel_unmatched.csv"), **to_kwargs
    )

    html_content = f"""
        <p>Hello,</p>
        <p>This is a weekly summary of matches between Flywheel sessions and REDCap records.</p>
        <br>
        <p>New matches since the last summary:</p>
        {new_matches_html}
        <br><br>
        <p>REDCap records that didn't match any Flywheel sessions:</p>
        {just_rc_html}
        <br><br>
        <p>Flywheel sessions that didn't match any REDCap records:</p>
        {just_fw_html}
        <br><br>
        <p>Best,</p>
        <p>WBHI Team</p>
    """
    csv_list = ["matches.csv", "redcap_unmatched.csv", "flywheel_unmatched.csv"]

    if test_run:
        recipients = [ADMIN_EMAIL]
    else:
        recipients = EMAIL_DICT[site]

    send_email(
        "Weekly WBHI Summary",
        html_content,
        config["gmail_address"],
        recipients,
        config["gmail_password"],
        [os.path.join(csv_path, basename) for basename in csv_list],
    )

    if site != "admin" and not test_run and not new_matches_df_copy.empty:
        for index, ses_id in new_matches_df_copy["ses_id"].items():
            session = client.get_session(ses_id)
            if "email" not in session.tags:
                session.add_tag("email")
            else:
                log.error("Session %s already has an 'email' tag.", session.id)


def main():
    gtk_context.init_logging()
    gtk_context.log_config()

    redcap_api_key = config["redcap_api_key"]
    redcap_project = Project(REDCAP_API_URL, redcap_api_key)
    pre_deid_project = client.lookup("wbhi/pre-deid")

    new_matches_df = create_new_matches_df(pre_deid_project)
    just_fw_df = create_just_fw_df()
    just_rc_df = create_just_rc_df(redcap_project)

    failed_jobs_df = create_failed_jobs_df()
    long_interval_df = create_long_interval_df(pre_deid_project)

    log.info("Sending emails to admin...")
    send_wbhi_email(
        new_matches_df, just_rc_df, just_fw_df, "admin", test_run=config["test_run"]
    )

    log.info("Sending emails to individual sites...")
    for site in SITE_LIST:
        send_wbhi_email(
            new_matches_df, just_rc_df, just_fw_df, site, test_run=config["test_run"]
        )


if __name__ == "__main__":
    with flywheel_gear_toolkit.GearToolkitContext() as gtk_context:
        config = gtk_context.config
        client = gtk_context.client

        main()
