## ADDED Requirements

### Requirement: User-provided YOLO TXT data is read through an explicit dataset configuration
The system SHALL load a configured user-provided detection dataset with one YOLO TXT label file per image. Each non-empty label row SHALL follow `class_id x_center y_center width height` with normalized box coordinates. The configuration SHALL define image and label locations, class order, and train/validation split. The loader SHALL validate label values and references, and SHALL NOT modify source data or download data implicitly.

#### Scenario: Valid dataset is loaded
- **WHEN** the configured image, label, class, and split files follow the documented YOLO TXT contract
- **THEN** the loader returns paired samples with image pixels, class IDs, and boxes in the training pipeline's documented coordinate convention

#### Scenario: Invalid or incomplete annotations are found
- **WHEN** a label has an out-of-range class ID, invalid normalized coordinates, or a required image/label pairing is missing
- **THEN** the loader reports the affected path and reason before training proceeds

#### Scenario: An image contains no objects
- **WHEN** a configured image has an empty label file
- **THEN** the loader represents it as a valid background image without changing the source file

#### Scenario: YOLOv1 grid cells contain multiple object centers
- **WHEN** multiple ground-truth centers map to the same YOLOv1 grid cell
- **THEN** target construction supervises the largest-area object in that cell and records the number of ignored objects, while dataset evaluation retains every original annotation

### Requirement: Version-specific training recipes run through a shared training entry point
The system SHALL provide one understandable training entry point for shared execution behavior while allowing each method's training recipe to specify its actual optimizer, schedule, augmentation, input size, batch size, seed, and version-specific objective. Each recipe SHALL record its source and disclose adaptations made for the common user dataset.

#### Scenario: YOLOv1 is trained using its configured recipe
- **WHEN** the user selects the YOLOv1 recipe and configured dataset
- **THEN** the shared trainer executes the recipe's values and keeps YOLOv1-specific model and objective behavior in its version implementation

#### Scenario: Two methods are compared on the common dataset
- **WHEN** the user runs two method recipes against the same configured split and class order
- **THEN** each run uses the same source dataset contract while retaining its own declared training settings

### Requirement: Dataset evaluation uses a shared detection-result contract
The system SHALL convert each method's predictions at an explicit boundary to `xyxy` pixel coordinates, class ID, and confidence, and SHALL evaluate against the configured validation split with the same documented metric definitions. The initial evaluator SHALL report AP@0.5 and AP@[0.5:0.95] and record the input size, preprocessing, confidence threshold, and NMS settings used for each run.

#### Scenario: YOLOv1 validation is evaluated
- **WHEN** the user runs evaluation for a trained YOLOv1 checkpoint
- **THEN** the evaluator reports per-class AP and aggregate metrics for the configured validation split using the shared prediction contract

#### Scenario: Methods with different native outputs are evaluated
- **WHEN** a method returns a native prediction format different from `xyxy` pixel coordinates
- **THEN** its explicit adapter converts predictions to the common contract before shared metric calculation

#### Scenario: Backend parity is checked
- **WHEN** the user compares PyTorch with an exported or converted backend
- **THEN** that numerical or detection parity result is recorded separately from dataset-level AP metrics

### Requirement: Each experiment records enough provenance to reproduce its result
The system SHALL save an immutable per-run snapshot of the selected recipe, dataset configuration and split identity, class order, random seed, code revision when available, environment details, evaluation settings, and reported metrics. Run outputs and checkpoints SHALL be stored separately from source data and source code.

#### Scenario: A training and evaluation run completes
- **WHEN** the run produces a checkpoint and validation result
- **THEN** the output directory contains the configuration snapshot and result metadata that identify how the checkpoint and metrics were produced

#### Scenario: An experiment is compared later
- **WHEN** the user reviews results from two runs
- **THEN** the saved metadata exposes differences in recipe and evaluation conditions alongside the metrics

### Requirement: Training, evaluation, and inference have explicit commands
The system SHALL expose documented commands from the repository root for training, evaluating, and running inference with YOLOv1. Importing project modules SHALL NOT start training, download files, or modify user data.

#### Scenario: A user follows the documented workflow
- **WHEN** the user runs the documented command with local data and weights configured
- **THEN** the selected operation executes explicitly and writes generated artifacts only to its configured output directory
