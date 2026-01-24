from pathlib import Path
import utils

def batch_analyze_eventdata(root_dir: Path, create_master: bool = False): # parent directory containing all videos that you want to include in the master file
    # Recursively loop through dir and find directories that contain an _eventdata.csv file and process them
    mvt_data_files = list(root_dir.rglob("*_eventdata.csv"))
    for mvt_data_file in mvt_data_files:        
        # Take raw _eventdata.csv file (with pixel area) and calculate volumes, format, etc
        df, metadata, video_name = utils.process_event_data(mvt_data_file.parent)
        # Take _eventdata_formatted.csv file and summarize each mouse's behavior in run
        utils.summarize_event_data(df, metadata, mvt_data_file.parent, video_name)

    # Collect all run_summaries to create master file
    if create_master:
        utils.create_master_file(root_dir, mvt_data_files)

def create_group_summaries(parent_dir: Path, group_folder_names: list):
    for group in group_folder_names:
        group_dir = parent_dir / group
        utils.create_group_summary_bymouse(group_dir)
        utils.create_group_summary_byrun(group_dir)


# ######################################################################################################
# ######################################################################################################

# batch_analyze_eventdata(Path(r"C:\Users\rlee21\Documents\MVT\MVTT_testing"), True)
# batch_analyze_eventdata(Path(r"C:\Users\Richard\_Vork\MVT\ERa_KO"), True)
batch_analyze_eventdata(Path(r"C:\Users\rlee21\Documents\MVT\Penk_FRT_Gq"), True)


# create_group_summaries(
#     Path(r"C:\Users\rlee21\OneDrive - Beth Israel Lahey Health\Verstegen-lab Shared OneDrive\ERa Project\MVT\MVT_files\averaged_runs"), 
#     ["AdultF_combined", "GSAdultM", "GSTeenM", "Machado Old Females", "MenoF_bymouse", "Nulliparous", "OldMales", "Parous", "TeenF_combined", "VCD"]
#     )

# utils.create_group_summary_bymouse(Path(r"C:\Users\rlee21\OneDrive - Beth Israel Lahey Health\Verstegen-lab Shared OneDrive\ERa Project\MVT\MVT_files\averaged_runs\VCD"))
# utils.create_group_summary_byrun(Path(r"C:\Users\rlee21\OneDrive - Beth Israel Lahey Health\Verstegen-lab Shared OneDrive\ERa Project\MVT\MVT_files\averaged_runs\VCD"))
