## ADDED Requirements

### Requirement: Version-specific implementation remains inspectable
The system SHALL keep each YOLO version's model structure, version-specific training objective, label assignment, decoding, and postprocessing in clearly named version-specific modules. Shared orchestration SHALL call those modules through explicit Python interfaces without hiding behavioral differences behind implicit version switches.

#### Scenario: Reader traces YOLOv1 tensor flow
- **WHEN** a reader follows the YOLOv1 implementation from input through prediction
- **THEN** the model and version-specific objective expose documented tensor shapes and the code identifies where decoding and postprocessing occur

#### Scenario: A later method changes a version-specific component
- **WHEN** a new paper version or custom method introduces a distinct head, assignment rule, or loss
- **THEN** that behavior is implemented in an identifiable method boundary and existing version behavior remains understandable

### Requirement: Version documentation connects papers to code
Each implemented version SHALL have one primary Chinese-language document that records the paper and official-source provenance, core idea, changes from earlier methods, major tensor flow, implementation files, run instructions, data and weight requirements, deviations, and validation status.

#### Scenario: Reader checks a claimed reproduction detail
- **WHEN** a reader compares a documented method with its code
- **THEN** the document identifies whether the detail comes from the paper, official code, or a disclosed project choice

#### Scenario: A run command is documented
- **WHEN** a version document provides a training, evaluation, or inference command
- **THEN** the command and its configuration path correspond to an implemented entry point, or are explicitly marked as not yet run

### Requirement: New versions and custom methods have an explicit extension path
The system SHALL allow a new YOLO paper version to be added as a version-specific implementation and a custom method to be represented by the components it changes. The system MUST NOT require a general plugin registry or force unrelated method behavior into an opaque shared model class.

#### Scenario: A new YOLO paper version is added
- **WHEN** an implementation is added for a new paper version
- **THEN** it has a distinct version boundary and documentation while using shared training and evaluation orchestration where behavior is genuinely common

#### Scenario: A custom method is added
- **WHEN** a custom method modifies only selected components of an existing method
- **THEN** its configuration and changed components are identifiable without rewriting unrelated version implementations
