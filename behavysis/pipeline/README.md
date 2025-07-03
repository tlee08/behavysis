# Pipeline

Orchestrates high-level experiment and project workflows.

- `experiment.py`: Experiment-level logic
- `project.py`: Project-level orchestration

## Usage

Used to coordinate the full analysis pipeline, managing data flow between processes and modules.

```mermaid
flowchart TD
    A[Project Config] --> B[Project Orchestration]
    B --> C[Experiment Logic]
    C --> D[Pipeline Processes]
    D --> E[Results]
```
