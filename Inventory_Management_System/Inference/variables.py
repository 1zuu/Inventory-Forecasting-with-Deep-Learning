train_csv_path = 'data/train.csv'
test_csv_path = 'data/test.csv'
model_weights = 'weights/inventory_forecasting_model.h5'
model_converter = "weights/inventory_forecasting_model.tflite"
minmax_scaler_weights = 'weights/minmax_scaler_weights.sav'
loss_img = "visualization/loss_comparison.png"
confusion_matrix_img = "visualization/confusion_matrix.png"


input_columns = ['datetime','store','item']
output_columns = ['sales']

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
batch_size = 500
num_epoches = 10
validation_split = 0.08


# Sample input

sample_input = {
        'datetime' : '2018-02-05',
        'store'    : '3',
        'item'     : '34'
            }