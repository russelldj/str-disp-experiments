import os
from pathlib import Path

# Important folders
PROJECT_ROOT = Path(os.path.abspath(""), "..", "..", "..").resolve()
SCRATCH_ROOT = Path(Path.home(), "scratch", "organized_str_disp_MVMT_experiments")

# Ground truth information
LABELS_FILENAME = Path(
    PROJECT_ROOT, "field_ref", "crowns_drone_w_field_data_no_QUEV.geojson"
)
LABELS_COLUMN = "species_observed"

# Conversion between short and long names
LONG_SITE_NAME_DICT = {"valley": "ValleyA", "chips": "ChipsB", "delta": "DeltaB"}

# Python utilities
MMSEG_UTILS_PYTHON = "/ofo-share/repos-david/conda/envs/mmseg-utils/bin/python"
MMSEG_PYTHON = "/ofo-share/repos-david/conda/envs/openmmlab/bin/python"

FOLDER_TO_CITYSCAPES_SCRIPT = "/ofo-share/repos-david/mmsegmentation_utils/dev/dataset_creation/folder_to_cityscapes.py"
VIS_PREDS_SCRIPT = "/ofo-share/repos-david/mmsegmentation_utils/dev/visualization/visualize_semantic_labels.py"
TRAIN_SCRIPT = "/ofo-share/repos-david/mmsegmentation/tools/train.py"
INFERENCE_SCRIPT = "/ofo-share/repos-david/mmsegmentation/tools/inference.py"


def get_IDs_to_labels(with_ground=False):
    IDs_to_labels = {
        0: "ABCO",
        1: "CADE",
        2: "PILA",
        3: "PIPO",
        4: "PSME",
        5: "SNAG",
    }
    if with_ground:
        IDs_to_labels[7] = "ground"

    return IDs_to_labels


def get_mesh_filename(short_model_name):
    CHIPS_MESH_FILENAME = Path(
        PROJECT_ROOT,
        "per_site_processing",
        short_model_name,
        "02_photogrammetry",
        "exports",
        "meshes",
        "ChipsB-120m_20230309T0502_w-mesh_w-80m_20231114T2219.ply",
    )
    # The mesh exported from Metashape
    DELTA_MESH_FILENAME = Path(
        PROJECT_ROOT,
        "per_site_processing",
        short_model_name,
        "02_photogrammetry",
        "exports",
        "meshes",
        "DeltaB-120m_20230310T1701_w-mesh_w-80m_20231117T1746.ply",
    )
    # The mesh exported from Metashape
    VALLEY_MESH_FILENAME = Path(
        PROJECT_ROOT,
        "per_site_processing",
        short_model_name,
        "02_photogrammetry",
        "exports",
        "meshes",
        "ValleyA-120m_20230323T0515_w-mesh.ply",
    )
    MESH_FILENAME_DICT = {
        "chips": CHIPS_MESH_FILENAME,
        "delta": DELTA_MESH_FILENAME,
        "valley": VALLEY_MESH_FILENAME,
    }

    return MESH_FILENAME_DICT[short_model_name]


def get_camera_filename(short_model_name):
    # The camera file exported from Metashape
    CHIPS_CAMERAS_FILENAME = Path(
        PROJECT_ROOT,
        "per_site_processing",
        short_model_name,
        "02_photogrammetry",
        "exports",
        "cameras",
        "ChipsB-120m_20230309T0502_w-mesh_w-80m_20231114T2219_abs_paths.xml",
    )
    # The camera file exported from Metashape
    DELTA_CAMERAS_FILENAME = Path(
        PROJECT_ROOT,
        "per_site_processing",
        short_model_name,
        "02_photogrammetry",
        "exports",
        "cameras",
        "DeltaB-120m_20230310T1701_w-mesh_w-80m_20231117T1746_abs_paths.xml",
    )
    # The camera file exported from Metashape
    VALLEY_CAMERAS_FILENAME = Path(
        PROJECT_ROOT,
        "per_site_processing",
        short_model_name,
        "02_photogrammetry",
        "exports",
        "cameras",
        "ValleyA-120m_20230323T0515_w-mesh_w-90m_20240212.xml",
    )

    CAMERAS_FILENAME_DICT = {
        "chips": CHIPS_CAMERAS_FILENAME,
        "delta": DELTA_CAMERAS_FILENAME,
        "valley": VALLEY_CAMERAS_FILENAME,
    }

    return CAMERAS_FILENAME_DICT[short_model_name]


def get_DTM_filename(short_model_name):
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        short_model_name,
        "02_photogrammetry",
        "exports",
        "dtms",
        f"{short_model_name}.tif",
    )


def get_image_folder(short_model_name):
    long_model_name = LONG_SITE_NAME_DICT[short_model_name]
    # The image folder used to create the Metashape project
    IMAGE_FOLDER = f"/ofo-share/str-disp_drone-data-partial/str-disp_drone-data_imagery-missions/{long_model_name}"
    return IMAGE_FOLDER


def get_oblique_images_folder(short_model_name):
    return {
        "chips": "/ofo-share/str-disp_drone-data-partial/str-disp_drone-data_imagery-missions/ChipsB/ChipsB_80m_2021_complete",
        "delta": "/ofo-share/str-disp_drone-data-partial/str-disp_drone-data_imagery-missions/DeltaB/DeltaB_80m",
        "valley": "/ofo-share/str-disp_drone-data-partial/str-disp_drone-data_imagery-missions/ValleyA/ValleyA_90m",
    }[short_model_name]


def get_labeled_mesh_filename(short_model_name):
    ## Define the intermediate results
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        short_model_name,
        "03_training_data",
        "labeled.ply",
    )


def get_render_scratch_folder(short_model_name):
    # Where to save the rendering label images
    return Path(
        SCRATCH_ROOT,
        "per_site_processing",
        short_model_name,
        "03_training_data",
        "renders",
    )


def get_render_folder(short_model_name):
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        short_model_name,
        "03_training_data",
        "renders",
    )


def get_images_near_labels_folder(short_model_name):
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        short_model_name,
        "03_training_data",
        "images_near_labels",
    )


def get_training_sites_str(training_sites):
    return "_".join(training_sites)


def get_training_data_scratch_folder(training_sites):
    training_sites_str = get_training_sites_str(training_sites)
    return Path(
        SCRATCH_ROOT,
        "models",
        "multi_site",
        training_sites_str,
    )


def get_aggregated_labels_folder(training_sites):
    training_data_folder = get_training_data_scratch_folder(training_sites)
    return Path(training_data_folder, "inputs", "labels")


def get_aggregated_images_folder(training_sites):
    training_data_folder = get_training_data_scratch_folder(training_sites)
    return Path(training_data_folder, "inputs", "images")


def get_work_dir(training_sites):
    training_data_folder = get_training_data_scratch_folder(training_sites)
    return Path(training_data_folder, "work_dir")


def get_inference_image_folder(site_name):
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        site_name,
        "03_training_data",
        "images_near_labels",
    )


def get_prediction_folder(prediction_site, training_sites):
    training_sites_str = get_training_sites_str(training_sites=training_sites)
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        prediction_site,
        "04_model_preds",
        f"{training_sites_str}_MVMT_model",
    )


def get_predicted_vector_labels_filename(prediction_site):
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        prediction_site,
        "05_processed_predictions",
        f"{prediction_site}_80m_chips_model.geojson",
    )


def get_numpy_export_faces_texture_filename(prediction_site):
    NUMPY_EXPORT_FACES_TEXTURE_FILE = Path(
        PROJECT_ROOT,
        "per_site_processing",
        prediction_site,
        "05_processed_predictions",
        f"{prediction_site}_80m_chips_model.npy",
    )


def get_numpy_export_cf_filename(prediction_site_name):
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        prediction_site_name,
        "05_processed_predictions",
        f"{prediction_site_name}_MVMT_confusion_matrix.npy",
    )
