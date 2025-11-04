# Ai-research-project
Here is the translation of the general explanation for the file `ashob1 (1).ipynb`:

**Main Project Goal:**
The primary goal of this project is image encryption using chaotic signals. The key innovation is the use of a "Multi-Level Validation" system based on deep learning (with TensorFlow/Keras) to ensure that the chaotic signal used for the encryption key genuinely possesses chaotic and secure properties.

**Main Code Components:**

1.  **`ChaoticSystemsGenerator`:**
    * This class generates 1D time-series signals from various chaotic systems (like Lorenz, Rossler, Chua, Henon, Logistic).
    * It also creates non-chaotic signals (like sine waves, noise, or linear) to be used as "invalid" training data.
    * It calculates metrics to quantify the chaotic nature of the signals, such as the Lyapunov exponent, correlation dimension, and entropy.

2.  **`MultiLevelChaoticValidator` (and `HyperparameterOptimizer`):**
    * This is the central component of the project and acts as a classifier.
    * It uses three different neural network models to analyze the signal at different scales:
        * **`build_pixel_level_network`:** (Pixel Level) A 1D Convolutional Neural Network (1D CNN) for small signal windows.
        * **`build_block_level_network`:** (Block Level) A combination of 1D CNN and LSTM for medium-sized windows.
        * **`build_global_level_network`:** (Global Level) A combination of 1D CNN, LSTM, and an Attention mechanism to analyze the entire signal.
    * An **`build_ensemble_model`** combines the results from these three levels to make the final decision on whether the signal is "valid" (chaotic).
    * The `HyperparameterOptimizer` class is also included to optimize the parameters of these networks (like the number of filters or the dropout rate).

3.  **`ImageEncryptionSystem`:**
    * This class uses the `ChaoticSystemsGenerator` to create a signal.
    * It then passes the signal to the `MultiLevelChaoticValidator` to confirm its validity.
    * If the signal is validated, it is converted into an encryption key, and the image is encrypted using a bitwise XOR operation.

4.  **`VisualizationSystem`:**
    * A helper class for plotting various results using Matplotlib and Seaborn.
    * This class plots items such as the model's training history, phase-space diagrams of the chaotic signals, and a security analysis of the encrypted image (including histograms, entropy, pixel correlation, NPCR, and UACI).

**Execution and Output:**
The portion of the code that was executed (cell 14) first trains all levels of the validation model using both chaotic and non-chaotic signals. The output indicates that the models were trained successfully (for example, the global-level validation accuracy reached 100%, and the block-level reached 98.75%).

After training, the system was tested on a sample signal and
correctly identified it as "Valid: True" (with a final score of 0.9734). Finally, the code loads a test image (named `encryption_analysis_lena_pattern.png`), successfully encrypts and decrypts it, and reports strong security metrics (such as NPCR 99.4% and UACI 8.05%).
