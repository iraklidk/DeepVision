# Cat vs Dog Classification — Technical Report

> Complete this after running the experiments and replace the bracketed fields with your own results.

## 1. Problem statement
The task is to classify an input image as either Cat or Dog. This is a binary image-classification problem.

## 2. Dataset and labels
- Dataset: Oxford-IIIT Pet
- Source split used: `trainval`
- Total images: [record actual number]
- Cat images: [record count]
- Dog images: [record count]
- Label mapping: Cat = 0; Dog = 1
- Split sizes: [train], [validation], [test]
- Split seed: 42

## 3. Preprocessing

Images are resized and cropped to `224 × 224` pixels, the input size used by ResNet-18. During training, random resized cropping and random horizontal flipping are applied to introduce variation in the images and help reduce overfitting.

Images are converted into PyTorch tensors and normalized using the ImageNet channel means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. This normalization matches the preprocessing used for the pretrained model.

**Data augmentation is applied only to the training set.** Validation and test images use resizing, center cropping, tensor conversion, and ImageNet normalization without random augmentation. This keeps evaluation consistent.

## 4. Model Architecture

The project uses a ResNet-18 convolutional neural network pretrained on ImageNet. The pretrained layers extract visual features, such as edges, textures, shapes, and more complex patterns.

The original model's final fully connected (`fc`) layer has 512 input features and 1,000 output classes. For this binary classification task, it is replaced with `nn.Linear(512, 2)`, producing two logits: one for Cat and one for Dog.

**Logits** are raw, unnormalized scores for the two classes. The predicted class is selected using `argmax`: `0` represents Cat and `1` represents Dog. During training, the logits are passed directly to the weighted cross-entropy loss function.

Two experiments are conducted:

- **Baseline:** The pretrained layers are frozen, and only the final `fc` layer is trained.
- **Fine-tuning:** The final ResNet block (`layer4`) is unfrozen and trained alongside the `fc` layer, allowing more of the pretrained model to adapt to the Cat vs Dog classification task.

## 5. Training configuration
| Setting | Baseline | Fine-tuning experiment |
|---|---|---|
| Trainable layers | Final `fc` layer | `layer4` and `fc` |
| Epochs | 5 | 5 |
| Batch size | 32 | 32 |
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.001 |
| Loss | Weighted cross-entropy | Weighted cross-entropy |
| Device | CPU | CPU |
| Training time | 518.1 seconds | 545.6seconds |

## 6. Results
Detailed evaluation results, including test metrics and classification reports, are saved in `outputs/baseline/` and `outputs/fine_tune/` These files are generated automatically when the evaluation script is run.

| Metric | Frozen baseline | Fine-tuned model |
|---|---:|---:|
| Test accuracy | 0.989 | 0.987 |
| Cat precision | 0.994 | 0.983 |
| Cat recall | 0.971 | 0.977 |
| Cat F1 | 0.982 | 0.98 |
| Dog precision | 0.986 | 0.989 |
| Dog recall | 0.997 | 0.991 |
| Macro F1 | 0.987 | 0.985 |

## 7. Error Analysis

The model's predictions are saved in `outputs/baseline/test_predictions.csv` and `outputs/fine_tune/test_predictions.csv`. The confusion matrices and classification reports are also saved in the corresponding output directories.

Error analysis involves comparing the predicted labels with the true labels to identify misclassified images. Possible reasons for incorrect predictions include unusual animal poses, distracting backgrounds, poor image quality, partial visibility, or images in which the animal is difficult to identify.

The baseline model achieved a test accuracy of 98.91%. Its classification report shows that Cat recall was lower than Dog recall, indicating that some cat images were misclassified. The saved predictions can be inspected to identify specific mistakes and investigate possible causes.

## 8. Optimization and Efficiency

The project compares a frozen ResNet-18 baseline with a fine-tuned version in which the final ResNet block (`layer4`) is also trainable.

The baseline training took approximately 518.1 seconds, while fine-tuning took approximately 545.6 seconds on a CPU. Fine-tuning therefore required approximately 27.5 additional seconds.

The baseline achieved a best validation loss of 0.0361, while the fine-tuned model achieved a best validation loss of 0.0297. However, a lower validation loss does not necessarily mean better test performance.

Checkpoint sizes and inference times should be measured on the same computer for both models. The comparison should use the same input images and measurement procedure. No reduction in model size or inference time is claimed unless supported by actual measurements.

## 9. Demo

The project includes a Streamlit application that allows users to classify their own images as Cat or Dog.

To launch the application, run the following command from the project root:

```bash
streamlit run app.py
```