train_csv_path = 'data/train.csv'
test_csv_path = 'data/test.csv'
food_csv_path = 'data/Foods.csv'
model_weights = 'weights/inventory_forecasting_model.h5'
model_converter = "weights/inventory_forecasting_model.tflite"
minmax_scaler_weights = 'weights/minmax_scaler_weights.sav'
confusion_matrix_img = "visualization/confusion_matrix.png"


input_columns = ['datetime','store','item']
output_columns = ['sales']

port = 5000
host = '0.0.0.0'

seed = 42
n_stores = 10 
n_items = 50

dense1 = 1024
dense2 = 512
dense3 = 256
dense4 = 64
dense_out = 1
keep_prob = 0.3

num_classes = 13

learning_rate = 0.0001
batch_size = 128
num_epoches = 10
validation_split = 0.05


# Sample input

sample_input = {
        "datetime" : "2018-02-05",
        "store"    : "3",
        "food"     : "Fish Pastry"
            }


model_results_img = 'visualization/model_results.png'
error_analysis_img = 'visualization/error_analysis.png'
loss_comparison_img = "visualization/loss_comparison.png"