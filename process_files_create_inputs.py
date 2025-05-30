from utils import preprocess_dataset


if __name__ == '__main__':
    # Create input files (along with word map)
    # preprocess_dataset(dataset_name='flickr8k', 
    #                    split_spec='/home/mihai/workspace/data/Flickr8k/captions.txt',         
    #                    image_dir='/home/mihai/workspace/data/Flickr8k/Images',               
    #                    caps_per_img=5,                           
    #                    min_freq=5,                                
    #                    output_dir='/home/mihai/workspace/output_data/Flickr8k',
    #                    max_length=50 )
    
    # preprocess_dataset(dataset_name='flickr30k', 
    #                    split_spec='/home/mihai/workspace/data/Flickr30k/captions.txt',         
    #                    image_dir='/home/mihai/workspace/data/Flickr30k/Images',               
    #                    caps_per_img=5,                           
    #                    min_freq=5,                                
    #                    output_dir='/home/mihai/workspace/output_data/Flickr30k',
    #                    max_length=50 )
    
    preprocess_dataset(dataset_name='coco', 
                       split_spec='/home/mihai/workspace/data/Coco/annotations',         
                       image_dir='/home/mihai/workspace/data/Coco/Images',               
                       caps_per_img=5,                           
                       min_freq=5,                                
                       output_dir='/home/mihai/workspace/output_data/Coco',
                       max_length=50 )