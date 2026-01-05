### How to test KINESIS with text-prompted synthetic motions

1. **Clone the modified MDM repository (original credit to Guy Tevet)**:
    ```bash
    git clone https://github.com/merc-s/motion-diffusion-model.git

    cd imitation-learning/motion-diffusion-model
    ```

2. **Follow the installation instructions**:
    ```bash
    pip install moviepy
    pip install git+https://github.com/openai/CLIP.git

    bash prepare/download_smpl_files.sh
    bash prepare/download_glove.sh
    bash prepare/download_t2m_evaluators.sh

    cd ..
    git clone https://github.com/EricGuo5513/HumanML3D.git
    unzip ./HumanML3D/HumanML3D/texts.zip -d ./HumanML3D/HumanML3D/
    cp -r HumanML3D/HumanML3D motion-diffusion-model/dataset/HumanML3D
    cd motion-diffusion-model
    ```


3. **Download the pre-trained model**:
    [Download link](https://drive.google.com/file/d/1cfadR1eZ116TIdXK7qDX1RugAerEiJXr/view)
    
    Unzip the folder and place the files `model000750000.pth` and `args.json` in a folder named `faster_save` in the root of the repository.
    ```bash
    mkdir faster_save
    ```

3. **Run the text-prompted motion synthesis**:
    ```bash
    python -m sample.generate --model_path ./faster_save/model000750000.pt --text_prompt "YOUR PROMPT HERE" --motion_length 3.3 --num_repetitions 10
    ```

4. **Create a folder for the generated motions**:
    ```bash
    mkdir generated_motions
    mv ./faster_save/samples_faster_save_000750000_seed10_YOUR_PROMPT_HERE/results.npy generated_motions/MOTION_NAME.npy
    ```

5. **Choose a repetition sample and generate .pkl file**:
    ```bash
    python -m visualize.numpy_to_pkl --input_path generated_motions/MOTION_NAME.npy --model mdm --rep 0
    ```

6. **Move to the Kinesis repository and generate the imitation-compatible .pkl file**:
    ```bash
    python utils/convert_data_mdm.py --input_path generated_motions/MOTION_NAME.pkl --output_dir data/t2m
    ```

7. **Run the model**:
    ```bash
    bash scripts/t2m.sh data/t2m/MOTION_NAME.pkl
    ```