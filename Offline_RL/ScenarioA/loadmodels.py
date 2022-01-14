import tensorflow as tf
from tensorflow import keras
import time
class models:
    def __init__(self):
        self.done=1
        #tf.compat.v1.disable_eager_execution()
    def load():
        models1=[]
        models2=[]
        models3=[]
        models4=[]
        print('begin load')
        T1=time.time()
        model1=keras.models.load_model("Zone1_models\modelt1")
        model2=keras.models.load_model("Zone2_models\modelt1")
        model3=keras.models.load_model("Zone3_models\modelt1")
        model4=keras.models.load_model("Zone4_models\modelt1")
        # model5=keras.models.load_model("Zone1_models\modelt2")
        # model6=keras.models.load_model("Zone2_models\modelt2")
        # model7=keras.models.load_model("Zone3_models\modelt2")
        # model8=keras.models.load_model("Zone4_models\modelt2")
        # model9=keras.models.load_model("Zone1_models\modelt3")
        # model10=keras.models.load_model("Zone2_models\modelt3")
        # model11=keras.models.load_model("Zone3_models\modelt3")
        # model12=keras.models.load_model("Zone4_models\modelt3")
        # model13=keras.models.load_model("Zone1_models\modelt4")
        # model14=keras.models.load_model("Zone2_models\modelt4")
        # model15=keras.models.load_model("Zone3_models\modelt4")
        # model16=keras.models.load_model("Zone4_models\modelt4")
        # model17=keras.models.load_model("Zone1_models\modelt5")
        # model18=keras.models.load_model("Zone2_models\modelt5")
        # model19=keras.models.load_model("Zone3_models\modelt5")
        # model20=keras.models.load_model("Zone4_models\modelt5")
        T2=time.time()
        print('end load',T2-T1)
        # for i in range(30):
        #     T1=time.time()
        #     tf.keras.backend.clear_session()
        #     models1.append(keras.models.load_model(f"Zone{1}_models\modelt{i+1}"))
        #     tf.keras.backend.clear_session()
        #     T2=time.time()
        #     print('end load model 1 of 4',T2-T1)
        #     models2.append(keras.models.load_model(f"Zone{2}_models\modelt{i+1}"))
        #     tf.keras.backend.clear_session()
        #     models3.append(keras.models.load_model(f"Zone{3}_models\modelt{i+1}"))
        #     tf.keras.backend.clear_session()
        #     models4.append(keras.models.load_model(f"Zone{4}_models\modelt{i+1}"))
        #     tf.keras.backend.clear_session()
        #     print('end load model 4 of 4',i)
        for i in range(30):
            models1.append(model1)
            models2.append(model2)
            models3.append(model3)
            models4.append(model4)
            # models1.append(model5)
            # models2.append(model6)
            # models3.append(model7)
            # models4.append(model8)
            # models1.append(model9)
            # models2.append(model10)
            # models3.append(model11)
            # models4.append(model12)
            # models1.append(model13)
            # models2.append(model14)
            # models3.append(model15)
            # models4.append(model16)
            # models1.append(model17)
            # models2.append(model18)
            # models3.append(model19)
            # models4.append(model20)
        return models1,models2,models3,models4

    @tf.function
    def serve1(x,model):
        prediction=model(x, training=False)
        return prediction

    @tf.function
    def serve2(x,model):
        prediction=model(x, training=False)
        return prediction

    @tf.function
    def serve3(x,model):
        prediction=model(x, training=False)
        return prediction

    @tf.function
    def serve4(x,model):
        prediction=model(x, training=False)
        return prediction