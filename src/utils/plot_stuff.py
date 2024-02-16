import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Set style and color palette
sns.set(style='darkgrid', palette='mako')

# Change the setting and put it in a dictionary
plot_settings = {
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'figure.dpi': 140,
    'axes.titlepad': 15,
    'axes.labelpad': 15,
    'figure.titlesize': 24,
    'figure.titleweight': 'bold',
}

# Use the dictionary variable to update the settings using matplotlib
plt.rcParams.update(plot_settings)


def plot_metrics(train_arr, val_arr, output_folder, logger, type='None'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_arr, label=f'Training {type}')
    plt.plot(val_arr, label=f'Validation {type}')
    plt.title(f'Training and Validation {type} Per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel(f'{type}')
    plt.legend()
    plt.savefig(f'{output_folder}/plot_train_val_{type}.png')
    logger.info(f'Saved images to: {output_folder}/plot_train_val_{type}.png')


def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Multiply by std and then add the mean
    return tensor


def plot_pred(inputs, labels, preds, output_folder, logger):
    num_images = len(inputs)
    cols = int(np.sqrt(num_images))
    rows = cols if cols**2 == num_images else cols + 1

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    # Adjust the space between images
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, ax in enumerate(axes.flatten()):
        if i < num_images:
            input = inputs[i]
            denormalized_input = denormalize(input.clone(), mean, std)
            img = denormalized_input.numpy().transpose((1, 2, 0))
            plt.imshow(img, cmap='gray')
            actual_label = 'Positive' if labels[i].item() == 1 else 'Negative'
            predicted_label = 'Positive' if preds[i].item(
            ) == 1 else 'Negative'
            ax.set_title(f'Actual: {actual_label}\nPredicted: {predicted_label}',
                         fontsize=10, backgroundcolor='white')
            ax.axis('off')  # Hide the axis
        else:
            ax.axis('off')  # Hide axis if no image

    plt.tight_layout()
    plt.savefig(f'{output_folder}/img_chest_pred.png')
    logger.info(f'Saved images to: {output_folder}/img_chest_pred.png')
    logger.info('Done training')


def plot_percentage_train_val(train_df, val_df, diseases, image_output='./'):
    # calculate the percentages of each disease in the train and validation sets
    train_percentages = train_df[diseases].mean() * 100
    val_percentages = val_df[diseases].mean() * 100

    # create a DataFrame that contains the calculated percentages
    data = {
        'Train': train_percentages,
        'Validation': val_percentages
    }
    percentage_df = pd.DataFrame(data)

    # reset index to make 'Disease' a column
    percentage_df = percentage_df.reset_index().rename(
        columns={'index': 'Disease'})

    # melt the DataFrame from wide format to long format for plotting
    percentage_df = percentage_df.melt(
        id_vars='Disease', var_name='Set', value_name='Percentage')

    # create a bar plot that compares the percentages of each disease in the train and validation sets
    plt.figure(figsize=(12, 8))
    sns.barplot(data=percentage_df, x='Percentage',
                y='Disease', hue='Set', alpha=1)
    plt.title('Comparison of Disease Percentages in Train and Validation Sets')
    plt.savefig(image_output)


def plot_number_patient_disease(df, diseases, image_output="./"):
    # What are the label counts for each disease?
    label_counts = df[diseases].sum().sort_values(ascending=False)
    # Plot the value counts
    plt.figure(figsize=(12, 8))
    sns.barplot(x=label_counts.values, y=label_counts.index)
    plt.xlabel('Number of Patients')
    plt.ylabel('Disease')
    plt.title('Number of Patients per Disease')
    plt.savefig(image_output)
