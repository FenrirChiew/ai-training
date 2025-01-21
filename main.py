from utils import *
from datetime import datetime
from sklearn.model_selection import train_test_split
from ai_model import AIModel


# Function to test the AI and display stats
def test_model(model, inputs, actions, count):
    total_predictions = len(inputs)
    correct_predictions = 0
    action_labels = ['Attack', 'Defend', 'Retreat']
    confusion_matrix = np.zeros((len(action_labels), len(action_labels)), dtype=int)

    for i, (test_input, actual_action) in enumerate(zip(inputs, actions)):
        output = model.forward_propagation(test_input)
        predicted_index = np.argmax(output)
        predicted_action = action_labels[predicted_index]
        actual_action_label = action_labels[actual_action]

        confusion_matrix[actual_action][predicted_index] += 1

        if predicted_index == actual_action:
            correct_predictions += 1

        print(
            f'Row {count + i + 1}: Test Input = {test_input} | Predicted Action = {predicted_action} | Actual Action = {actual_action_label}')

    # Display statistics
    accuracy = correct_predictions / total_predictions * 100
    print(f'\nAccuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})')

    print('\nConfusion Matrix:')
    print('       ' + '  '.join(action_labels))
    for i, row in enumerate(confusion_matrix):
        print(f'{action_labels[i]:<7} ' + ' '.join(f'{count:3}' for count in row))


# Load Combat Data
print('Loading Combat Data...')
combat_data = np.genfromtxt("combat_data.csv", delimiter=",", skip_header=1, dtype=int)
print(f'Total Rows: {combat_data.shape[0]}')
# Split the data into features and labels
X = combat_data[:, :-1]  # Features (all columns except the last one)
y = combat_data[:, -1]  # Labels (last column)

# Split into training (70%), validation (15%), and test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalize the data (using Min-Max normalization)
X_train_normalized = normalize_data(X_train)
X_val_normalized = normalize_data(X_val)
X_test_normalized = normalize_data(X_test)

# Define the ratio of training data
training_count = len(X_train_normalized)
print(f'Using first {training_count} Rows ({training_count / len(combat_data):.2%}) of the Combat Data for Training...')
training_inputs = X_train_normalized
training_actions = y_train

# Load action encoding (assumed to be a one-hot encoding in actions.csv)
actions_encoding = np.genfromtxt("actions.csv", delimiter=",", skip_header=1, dtype=int)
training_outputs = actions_encoding[training_actions]

# Initialize AI Model with different activation functions
# ['sigmoid', 'relu', 'leaky_relu']
input_size = training_inputs.shape[1]
hidden_size = max(5, input_size * 2)
output_size = actions_encoding.shape[1]
combat_ai = AIModel(input_size=training_inputs.shape[1],
                    hidden_size=10,
                    output_size=actions_encoding.shape[1],
                    activation_function='sigmoid')

# Training Loop
iterations = 100000  # Set a default high number of iterations
learning_rate = 0.001  # Default learning rate
max_gradient_norm = 1.0  # Default maximum allowed gradient norm for clipping
start = datetime.now()

combat_ai.train(training_inputs, training_outputs, validation_inputs=X_val_normalized, validation_outputs=y_val,
                iterations=iterations, learning_rate=learning_rate, max_gradient_norm=max_gradient_norm)

elapsed = datetime.now() - start
print(f'Training Completed. Elapsed: {elapsed}')

# Testing the model with the remaining data
test_inputs = X_test_normalized
test_actions = y_test

test_model(combat_ai, test_inputs, test_actions, training_count)

# Fine-tuning loop
while True:
    fine_tune = input("\nDo you want to fine-tune the model? (y to continue): ").strip().lower()

    if fine_tune != "y":
        print("Stopping fine-tuning process.")
        break

    # Ask the user for how many more iterations or if they want to change the learning rate
    try:
        additional_iterations = int(input("Enter additional iterations (default is 10000): ").strip() or "10000")
        new_learning_rate = input("Enter new learning rate (default is 0.1): ").strip()

        # Set default learning rate if no new value is provided
        new_learning_rate = float(new_learning_rate) if new_learning_rate else learning_rate

        new_max_gradient_norm = input("Enter new maximum allowed gradient norm (default is 5.0): ").strip()

        # Set default maximum allowed gradient norm if no new value is provided
        new_max_gradient_norm = float(new_max_gradient_norm) if new_max_gradient_norm else max_gradient_norm

        # Start fine-tuning
        print(
            f"Fine-tuning for {additional_iterations} iterations with learning rate {new_learning_rate} and maximum allowed gradient norm {new_max_gradient_norm}...")
        combat_ai.train(training_inputs, training_outputs, validation_inputs=X_val_normalized, validation_outputs=y_val,
                        iterations=additional_iterations, learning_rate=new_learning_rate,
                        max_gradient_norm=new_max_gradient_norm)

        print("Testing model after fine-tuning...")
        test_model(combat_ai, test_inputs, test_actions, training_count)

    except ValueError:
        print("Invalid input. Please enter valid numbers for iterations and learning rate.")
