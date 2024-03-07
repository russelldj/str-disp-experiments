import os
from pathlib import Path

# Important folders
# PROJECT_ROOT = Path(os.path.abspath(""), "..", "..", "..").resolve()
# SCRATCH_ROOT = Path(Path.home(), "scratch", "organized_str_disp_MVMT_experiments")

PROJECT_ROOT = Path(os.path.abspath("/ofo-share"), "scratch-derek", "organized-str-disp-MVMT-experiments_80m")
SCRATCH_ROOT = Path(os.path.abspath("/ofo-share"), "scratch-derek", "organized-str-disp-MVMT-experiments_80m_scratch")

# Ground truth information
LABELS_FILENAME = Path(
    PROJECT_ROOT, "field_ref", "crowns_drone_w_field_data_updated_nosnags.gpkg"
)
LABELS_COLUMN = "species_observed"

# Conversion between short and long names
LONG_SITE_NAME_DICT = {
    "valley": "ValleyA",
    "chips": "ChipsB",
    "delta": "DeltaB",
    "lassic": "Lassic",
}

# Python utilities
MMSEG_UTILS_PYTHON = "/ofo-share/repos-david/conda/envs/mmseg-utils/bin/python"
MMSEG_PYTHON = "/ofo-share/repos-david/conda/envs/openmmlab/bin/python"

FOLDER_TO_CITYSCAPES_SCRIPT = "/ofo-share/repos-david/mmsegmentation_utils/dev/dataset_creation/folder_to_cityscapes.py"
VIS_PREDS_SCRIPT = "/ofo-share/repos-david/mmsegmentation_utils/dev/visualization/visualize_semantic_labels.py"
TRAIN_SCRIPT = "/ofo-share/repos-david/mmsegmentation/tools/train.py"
INFERENCE_SCRIPT = "/ofo-share/repos-david/mmsegmentation/tools/inference.py"


TRAINING_IMGS_EXT = ".png"
INFERENCE_IMGS_EXT = ".png"
CHIP_SIZE = 3648
BATCH_SIZE = 2
INFERENCE_STRIDE = int(CHIP_SIZE / 2)

AGGREGATE_IMAGE_SCALE = 1

# Points less than this height (meters) above the DTM are considered ground
HEIGHT_ABOVE_GROUND_THRESH = 2
# The image is downsampled to this fraction for accelerated rendering
RENDER_IMAGE_SCALE = 1
# Cameras within this distance of the traing data are used in the rendering process
BUFFER_RADIUS_METERS = 50
# Downsample target
DOWNSAMPLE_TARGET = 1


def get_IDs_to_labels(with_ground=False):
    IDs_to_labels = {
        0: "ABCO",
        1: "CADE",
        2: "PILA",
        3: "PIPJ",
        4: "PSME",
    }
    if with_ground:
        IDs_to_labels[5] = "ground"

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
    # The mesh exported from Metashape
    LASSIC_MESH_FILENAME = Path(
        PROJECT_ROOT,
        "per_site_processing",
        short_model_name,
        "02_photogrammetry",
        "exports",
        "meshes",
        "Lassic-120m_20240213T0503_model_local.ply",
    )
    MESH_FILENAME_DICT = {
        "chips": CHIPS_MESH_FILENAME,
        "delta": DELTA_MESH_FILENAME,
        "lassic": LASSIC_MESH_FILENAME,
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
    # The camera file exported from Metashape
    LASSIC_CAMERAS_FILENAME = Path(
        PROJECT_ROOT,
        "per_site_processing",
        short_model_name,
        "02_photogrammetry",
        "exports",
        "cameras",
        "Lassic-120m_20240213T0503_w-mesh_w-80m_20240214_aligned.xml",
    )

    CAMERAS_FILENAME_DICT = {
        "chips": CHIPS_CAMERAS_FILENAME,
        "delta": DELTA_CAMERAS_FILENAME,
        "lassic": LASSIC_CAMERAS_FILENAME,
        "valley": VALLEY_CAMERAS_FILENAME,
    }

    return CAMERAS_FILENAME_DICT[short_model_name]


def get_mesh_transform_filename(site_name):
    if site_name in ("chips", "delta", "valley"):
        return get_camera_filename(site_name)
    elif site_name == "lassic":
        # Lassic was processed differently, so this hack is required
        return Path(
            "/ofo-share/str-disp_drone-data-partial/imagery-processed/outputs/120m-01/Lassic-120m_20240213T0503_cameras.xml"
        )
    else:
        raise ValueError(f"Site {site_name} not found")


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
        "lassic": "/ofo-share/str-disp_drone-data-partial/str-disp_drone-data_imagery-missions/Lassic/Lassic_80m",
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


def get_training_data_folder(training_sites, is_ortho, is_scratch):
    training_sites_str = get_training_sites_str(training_sites)
    ortho_or_mvmt_str = "ortho" if is_ortho else "MVMT"
    return Path(
        SCRATCH_ROOT if is_scratch else PROJECT_ROOT,
        "models",
        "multi_site",
        ortho_or_mvmt_str + "_" + training_sites_str,
    )


def get_aggregated_labels_folder(training_sites, is_ortho):
    training_data_folder = get_training_data_folder(
        training_sites, is_ortho=is_ortho, is_scratch=True
    )
    ortho_or_mvmt_str = "ortho" if is_ortho else "MVMT"
    return Path(training_data_folder, ortho_or_mvmt_str, "inputs", "labels")


def get_aggregated_images_folder(training_sites, is_ortho):
    training_data_folder = get_training_data_folder(
        training_sites, is_ortho=is_ortho, is_scratch=True
    )
    ortho_or_mvmt_str = "ortho" if is_ortho else "MVMT"
    return Path(training_data_folder, ortho_or_mvmt_str, "inputs", "images")


def get_work_dir(training_sites, is_ortho, is_scratch):
    training_data_folder = get_training_data_folder(
        training_sites, is_ortho=is_ortho, is_scratch=is_scratch
    )

    return Path(training_data_folder, "work_dir")


def get_inference_image_folder(site_name):
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        site_name,
        "03_training_data",
        "images_near_labels",
    )


def get_prediction_folder(prediction_site, training_sites, is_ortho):
    training_sites_str = get_training_sites_str(training_sites=training_sites)
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        prediction_site,
        "04_model_preds",
        f"{training_sites_str}_{'ortho' if is_ortho else 'MVMT'}_model",
    )


def get_predicted_output_base_file(prediction_site, training_sites):
    training_sites_str = get_training_sites_str(training_sites)
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        prediction_site,
        "05_processed_predictions",
        f"{prediction_site}_80m_{training_sites_str}_model",
    )


def get_predicted_vector_labels_filename(prediction_site, training_sites):
    return get_predicted_output_base_file(
        prediction_site=prediction_site, training_sites=training_sites
    ).with_suffix(".geojson")


def get_predicted_polygons_labels_filename(prediction_site, training_sites, is_ortho):
    base_file = get_predicted_output_base_file(
        prediction_site=prediction_site, training_sites=training_sites
    )
    return Path(
        str(base_file) + f"labeled_polygons_{'ortho' if is_ortho else 'MVMT'}.geojson"
    )


def get_numpy_export_faces_texture_filename(prediction_site, training_sites):
    return get_predicted_output_base_file(
        prediction_site=prediction_site, training_sites=training_sites
    ).with_suffix(".npy")


def get_numpy_export_cf_filename(prediction_site_name, training_sites, is_ortho):
    base_file = get_predicted_output_base_file(
        prediction_site=prediction_site_name, training_sites=training_sites
    )
    extension_str = f"_{'ortho' if is_ortho else 'MVMT'}_confusion_matrix.npy"
    return Path(str(base_file) + extension_str)


def get_numpy_export_confusion_matrix_file(inference_site, is_ortho):
    ortho_or_mvmt_str = "ortho" if is_ortho else "MVMT"
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        inference_site,
        "05_processed_predictions",
        f"{inference_site}_{ortho_or_mvmt_str}_confusion_matrix.npy",
    )


def get_figure_export_confusion_matrix_file(inference_site, is_ortho):
    ortho_or_mvmt_str = "ortho" if is_ortho else "MVMT"
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        inference_site,
        "05_processed_predictions",
        f"{inference_site}_{ortho_or_mvmt_str}_confusion_matrix.png",
    )


def get_training_raster_filename(training_site):
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        training_site,
        "02_photogrammetry",
        "exports",
        "orthos",
        f"{training_site}.tif",
    )


def get_training_chips_folder(training_site):
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        training_site,
        "03_training_data",
        f"ortho_chipped_images_{training_site}",
    )


def get_inference_raster_filename(inference_site):
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        inference_site,
        "02_photogrammetry",
        "exports",
        "orthos",
        f"{inference_site}.tif",
    )


def get_inference_chips_folder(inference_site):
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        inference_site,
        "04_model_preds",
        "ortho_chipped_images",
    )


def get_aggregated_raster_pred_file(training_sites, inference_site):
    training_sites_str = get_training_sites_str(training_sites=training_sites)
    return Path(
        PROJECT_ROOT,
        "per_site_processing",
        inference_site,
        "05_processed_predictions",
        f"{training_sites_str}_model_ortho_pred.tif",
    )
