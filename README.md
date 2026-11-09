# Sign ML

An offline Indian Sign Language (ISL) translator. It recognises signs from a webcam and shows the result as text and speaks it aloud.

Built for HerHack '26 (Echelon Institute of Technology).

## What it does

- Recognises ISL signs from a webcam using hand tracking (MediaPipe) and a Random Forest classifier (scikit-learn).
- Shows the recognised sign as text and speaks it.
- Runs offline on a PC (desktop app), and in a browser through a small local server.
- Each sign is a short movement, so the model looks at ~ 1s of hand movement, not a static single frame.

## How It Works

1. **Hand tracking.** MediaPipe finds 21 points on each hand in every webcam frame.
2. **Features.** The points are normalised so that hand position and size do not matter, then the last 30 frames (about one second) are joined into one sample. Looking at movement, not a single frame, lets the model handle signs that involve motion.
3. **Classification.** A Random Forest classifier (scikit-learn) predicts the sign from each sample.
4. **Output.** A sign is accepted when the model is confident and several predictions in a row agree. It is then shown on screen and spoken once. Lower your hand before signing the same letter again.

## Requirements

- Windows
- 64-bit Python 3.11
- A webcam
- Git (optional, for cloning)

## Setup for windows
You need **64bit Python 3.11** and Git. Then run:
git clone
https://github.com/LapisLazuli4/isl-hackathon.git
cd isl-hackathon
python -m venv isl_env
isl_env\Scripts\activate
pip install -r requirements.txt

The MediaPipe version is pinned to 0.10.14 because newer versions remove the API this project uses.

  
## Current accuracy

On a person whose recordings were **not** used in training, the model scored **about 70%**. This is an early prototype, not a finished translator.

## Project structure

- collect.py Record sign samples from a webcam (run by each team member)
- train.py Train the model from all CSV files in data/
- run.py Desktop live translator: webcam -> text and speech
- features.py Turns hand landmarks into numbers (shared by all scripts)
- server.py Local web server (FastAPI) for the browser version
- Browser front end (index.html, app.js)
- requirements.txt

## Usage
### 1. Record samples

Each contributor runs this on their own PC:

python collect.py --name YourName

- Press a letter to choose the sign.
- Press **SPACE** to start recording, and **SPACE** again to stop.
- Press **Esc** to quit.

Hold the sign steady for about a second, and vary your distance from the camera, hand position, speed, and lighting across takes. Recordings are saved to `data/YourName.csv`. Use the same name every time.

### 2. Train

Put every contributor's CSV file in the `data` folder, then run:

python train.py


It prints the number of samples per sign and per contributor, the accuracy on one person whose data was held out of training, a per-sign report, and a confusion matrix. It then saves the model, trained on everyone's data, to `isl_model.pkl`.

To choose which contributor is held out:

python train.py --test-contributor YourName


### 3. Run the translator

python run.py

Sign in front of the webcam. Press **C** to clear the text and **Q** to quit.

## Improving accuracy

- Record every sign from several different people. A model trained on one person learns that person, not the sign.
- Keep the number of samples roughly equal across signs.
- Record only the sign itself, not the hand moving into or out of position.
- Use the confusion matrix from `train.py` to find signs that are mixed up, then record more takes of them.
- Raise `CONFIDENCE` in `run.py` for fewer wrong signs, or lower it if correct signs are missed.

## Project structure

collect.py Record sign samples from a webcam
train.py Train the model from the CSV files in data/
run.py Live translator: webcam to text and speech
features.py Turns hand landmarks into numbers (shared by all scripts)
requirements.txt Python packages
LICENSE MIT licence (code)


## Data

Recordings are not included in this repository. The dataset is intended to be published separately under the CC BY 4.0 licence.

## Licence

The code is released under the MIT licence. See `LICENSE`.
