from eda_sda import eda_sda
def main():
    img_path = '../Datasets/chest_cancer_data/train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/000000 (6).png'
    folder_path = '../Datasets/chest_cancer_data/train'

    #EDA and SDA
    analysis_obj = eda_sda(img_path,folder_path,25)
    analysis_obj.init_generator()
    analysis_obj.stats()
    analysis_obj.display_img()
    analysis_obj.display_batch()

if __name__ == '__main__':
    main()