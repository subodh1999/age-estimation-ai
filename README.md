# Real-Time Age Estimation AI

A deep learning application that uses your webcam to estimate age in real-time, powered by a PyTorch MobileNetV2 model with a Streamlit frontend.

## Prerequisites

- **Python** 3.8 or higher
- A working **webcam**

## Project Files

Make sure all these files are in the same folder:

| File | Description |
|------|-------------|
| `app.py` | Main application code |
| `requirements.txt` | Python dependencies |
| `best_mobilenetv2.pth` | Trained model weights *(app will fail without this)* |

## How to Run

### 1. Open your Terminal

Open Command Prompt (Windows) or Terminal (Mac/Linux) and navigate to the project folder:

```bash
cd "path/to/your/project/folder"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the App

```bash
streamlit run app.py
```

The app will open automatically in your default browser. Allow webcam access when prompted.
