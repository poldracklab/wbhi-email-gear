#!/usr/bin/env python

import csv
import flywheel_gear_toolkit
import flywheel
import logging
import pandas as pd
from redcap import Project
from flywheel import (
    ProjectOutput,
    SessionListOutput,
    AcquisitionListOutput,
    Gear
)

log = logging.getLogger(__name__)

REDCAP_API_URL = "https://redcap.stanford.edu/api/"

def get_acq_df(container) -> pd.DataFrame:
    """Get unique labels for all acquisitions in the container.

    This is done using a single Data View which is more efficient than iterating through
    all acquisitions, sessions, and subjects. This prevents time-out errors in large projects.
    """

    columns = [
        'subject.id',
        'subject.label',
        'session.id',
        'session.info.BIDS',
        'acquisition.label',
        'file.id',
        'file.tags',
        'file.type'
    ]
    #filter = 'file.type=dicom,file.type=nifti'
    filter = ''

    builder = flywheel.ViewBuilder(
       label="Find all dicom files in the project.",
        container='acquisition',
        filename="*.*",
        match='all',
        process_files=False,
        include_ids=False,
        include_labels=False,
        filter=filter
    )
    for c in columns:
        builder.column(src=c)
   
    view = builder.build()
    return client.read_view_dataframe(view, container.id)

def get_deid_sessions(acq_df: pd.DataFrame) -> list:
    dcm_tags = {
            'file-metadata-importer-deid',
            'file-classifier-deid',
            'dcm2niix'
    }
    nii_tag = 'pydeface'

    # Skip subject if all sessions have been bidsified
    nonbids_df = acq_df.groupby('subject.id').filter(
        lambda x: x['session.info.BIDS'].isna().any()
    )
    
    # Skip sessions if any dicoms or niftis missing proper tags
    nii_df = nonbids_df[nonbids_df['file.type'] == 'nifti']
    nii_df = nii_df[nii_df['file.tags'].apply(lambda x: nii_tag in x)]
    filt_df = nonbids_df[nonbids_df['session.id'].isin(nii_df['session.id'])]
    dcm_df = nii_df[nii_df['file.type'] == 'dicom']
    dcm_df = dcm_df[dcm_df['file.tags'].apply(
        lambda x, tags=dcm_tags: dcm_tags.issubset(x)
    )]
    if not dcm_df.empty:
        filt_df = filt_df[filt_df['session.id'].isin(dcm_df['session.id'])]
    else:
        filt_df = filt_df[0:0]

    # Raise warning if some non-bidsified sessions are filtered out
    nonbids_ses_set = set(nonbids_df['subject.label'])
    filt_ses_set = set(filt_df['subject.label'])
    filt_out_set = nonbids_ses_set - filt_ses_set
    if filt_out_set:
        log.warning("The following subjects are not yet bidsified but are "
                    f"contain files that are missing necessary tags: \n{filt_out_set}"
        )
    return filt_df['session.id'].tolist()

def import_reproin_csv(deid_project: ProjectOutput) -> dict:
    with gtk_context.open_input('to_reproin_csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        return {row[0]: row[1] for row in csv_reader}

def main():
    gtk_context.init_logging()
    gtk_context.log_config()

    redcap_api_key = config["redcap_api_key"]
    redcap_project = Project(REDCAP_API_URL, redcap_api_key)
    redcap_data = redcap_project.export_records()
    
    #deid_project = client.lookup("wbhi/deid")
    deid_project = client.lookup("joe_test/deid_joe")
    acq_df = get_acq_df(deid_project)
    sessions = get_deid_sessions(acq_df)
    to_reproin_dict = import_reproin_csv(deid_project)
    breakpoint()

if __name__ == "__main__":
    with flywheel_gear_toolkit.GearToolkitContext() as gtk_context:
        config = gtk_context.config
        client = gtk_context.client
        
        main()

