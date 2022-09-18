from tensorboard import program

tracking_address = "C:/Users/yanir/PycharmProjects/masterThesis/ml_models/hparam_tuning/" # the path of your log file.

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    tb.launch()

# Please RUN in terminal
# tensorboard --logdir=ml_models/hparam_tuning