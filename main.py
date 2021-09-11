from eda_sda import eda_sda
from model import model
from pre_process import pre_processing

class main():

    def __init__(self):
        self.img_path = '../Datasets/chest_cancer_data/train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/000000 (6).png'
        self.train_path = '../Datasets/chest_cancer_data/train'
        self.valid_path = '../Datasets/chest_cancer_data/valid'
        self.test_path = '../Datasets/chest_cancer_data/test'
        self.batch_size = 25

    # Exploratory & Statistical Data Analysis
    def data_analysis(self):
        analysis_obj = eda_sda(self.img_path,self.train_path,self.batch_size)
        analysis_obj.init_generator()
        analysis_obj.stats()
        analysis_obj.display_img()
        analysis_obj.display_batch()

    # Model creation, training, & testing
    def cnn_model(self):
        epochs = 50
        target_size = (256,380)
        input_shape = (256,380,3)
        pool,kernel = (2,2),(3,3)
        train_batch_size,valid_batch_size,test_batch_size = 25,5,15

        dpp_obj = pre_processing(self.train_path,self.valid_path,
        self.test_path,target_size,train_batch_size,valid_batch_size,test_batch_size)

        train_gen,valid_gen,test_gen = dpp_obj.init_generators()
        model_obj = model(epochs,input_shape,pool,kernel,
                train_gen,valid_gen,test_gen)
        
        # Train and evaluate
        model_obj.results()
        model_obj.final_model()

    
if __name__ == '__main__':
    main_obj = main()
    # main_obj.data_analysis()
    main_obj.cnn_model()
