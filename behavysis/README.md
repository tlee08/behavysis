# Behavysis

This is the main package containing all core modules for the Behavysis behavioral analysis pipeline. Submodules include behavioral classification, dataframes for analysis, pipeline orchestration, processing steps, configuration models, utilities, and the viewer GUI.

## Structure

- `behav_classifier/`: Behavioral classification models and logic
- `df_classes/`: DataFrame-based analysis classes
- `pipeline/`: Project and experiment orchestration
- `processes/`: Individual processing steps
- `pydantic_models/`: Configuration and data validation models
- `scripts/`: Entry points and utility scripts
- `utils/`: General-purpose utilities
- `viewer/`: GUI and visualization components

See each submodule's README for more details.

```mermaid
flowchart LR
    %% File Nodes
    F4@{ shape: st-rect, label: "Novel Videos"}
    F5@{ shape: st-rect, label: "Keypoints"}
    F6@{ shape: st-rect, label: "Predicted Behaviour"}
    F7@{ shape: st-rect, label: "Scored Behaviour"}

    %% Model Nodes
    M0@{ shape: diamond, label: "DLC model" }
    M1@{ shape: diamond, label: "Behavysis model" }

    %% Connections
    F4 & M0 --> F5
    F5 & M1 --> F6
    F6 --> F7

    %% Classes for styling
    classDef FileNode fill:#fdf6e3,stroke:#333;
    classDef ModelNode fill:#b3e6ff,stroke:#333;

    %% Apply classes
    class F0,F1,F2,F3,F4,F5,F6,F7 FileNode;
    class M0,M1 ModelNode;
```

```mermaid
flowchart LR

    subgraph S0 [Keypoints Model]
        direction LR
        %% File Nodes
        F0@{ shape: st-rect, label: "Videos"}
        F1@{ shape: st-rect, label: "Labelled keypoints"}

        %% Model Nodes
        M0@{ shape: diamond, label: "DLC model" }

        %% Connections
        F0 & F1 --> M0
    end

    subgraph S1 [Behaviour Model]
        direction LR
        %% File Nodes
        F2@{ shape: st-rect, label: "Labelled keypoints"}
        F3@{ shape: st-rect, label: "Labelled behaviours"}

        %% Model Nodes
        M1@{ shape: diamond, label: "Behavysis model" }

        %% Connections
        F2 & F3 --> M1
    end

    %% File Nodes
    F4@{ shape: st-rect, label: "Novel Videos"}
    F5@{ shape: st-rect, label: "Keypoints"}
    F6@{ shape: st-rect, label: "Predicted Behaviour"}
    F7@{ shape: st-rect, label: "Scored Behaviour"}

    %% Connections
    F4 & M0 --> F5
    F5 & M1 --> F6
    F6 --> F7

    %% Classes for styling
    classDef FileNode fill:#fdf6e3,stroke:#333;
    classDef ModelNode fill:#b3e6ff,stroke:#333;

    %% Apply classes
    class F0,F1,F2,F3,F4,F5,F6,F7 FileNode;
    class M0,M1 ModelNode;
```

```mermaid
flowchart TD
    %% File Nodes
    F0@{ shape: st-rect, label: "0_configs<br>Stores how to process the experiment at each step<br>e.g. DLC model, cutoffs, etc." }
    F1@{ shape: st-rect, label: "1_raw_vid"}
    F2@{ shape: st-rect, label: "2_formatted_vid"}
    F3@{ shape: st-rect, label: "3_keypoints"}
    F4@{ shape: st-rect, label: "4_preprocessed"}
    F5@{ shape: st-rect, label: "5_features_extracted"}
    F6@{ shape: st-rect, label: "6_predicted_behavs"}
    F7@{ shape: st-rect, label: "7_scored_behavs"}
    F8@{ shape: st-rect, label: "8_analysis"}
    F9@{ shape: st-rect, label: "9_analysis_combined"}
    F10@{ shape: st-rect, label: "10_evaluate_vid"}

    %% Process Nodes
    P0@{ shape: pill, label: "down-sample" }
    P1@{ shape: pill, label: "keypoint tracking" }
    P2@{ shape: pill, label: "preprocess" }
    P3@{ shape: pill, label: "derive features for<br>behaviour classifier" }
    P4@{ shape: pill, label: "auto-classify<br>behaviours" }
    P5@{ shape: pill, label: "manual check<br>behaviours" }
    P6@{ shape: pill, label: "aggregate and summarise" }
    P7@{ shape: pill, label: "analyse, aggregate,<br>and summarise" }
    P8@{ shape: pill, label: "combine analysis into<br>single table" }
    P9@{ shape: pill, label: "make evaluation video" }

    %% Connections
    F1 --- P0 --> F2
    F2 --- P1 --> F3
    F3 --- P2 --> F4
    F4 --- P3 --> F5
    F5 --- P4 --> F6
    F2 & F4 & F6 --- P5 --> F7
    F7 --- P6 --> F8
    F4 --- P7 --> F8
    F8 --- P8 --> F9
    F2 & F9 --- P9 --> F10

    %% Classes for styling
    classDef FileNode fill:#fdf6e3,stroke:#333;
    classDef ProcessNode fill:#b3e6ff,stroke:#333;

    %% Apply classes
    class F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10 FileNode;
    class P0,P1,P2,P3,P4,P5,P6,P7,P8,P9 ProcessNode;
```
