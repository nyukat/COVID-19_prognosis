from torch.utils.data import Dataset
import torch

class RandomImageLoader(Dataset):
    """
    Class that generates random images and labels.
    Generated random images will have shape as specified in parameters["image_size"].
    By default it's (1024, 1024).
    """
    def __init__(self, parameters):
        self.labels = [torch.randint(low=0, high=2, size=(1, parameters["number_classes"])) for _ in range(parameters["data_num"])]
        self.imgs = [torch.rand(size=(3, parameters["image_size"][0], parameters["image_size"][1])) for _ in range(parameters["data_num"])]
        if parameters["drc"]:
            label_time = torch.rand((parameters["data_num"],1))*200
            label_event = (torch.rand((parameters["data_num"],1))>0.8).float()
            self.labels = torch.cat([label_time,label_event],dim = 1)
            
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        return index, img, label

    def __len__(self):
        return len(self.labels)
    
def discritize_time_array(time_array, buckets = [0.,3.,12.,24.,48.,72.,96.,144.,192.]):
    """
    The functions discritizes the time to adverse event into the given time buckets and returns binary labels for each bucket that indicates if patient survived the time bucket or not. It also outputs surv_labels which indicates if the patient was censored or not. If the patient was censored, surv_labels also indicates when the patient was censored.
    """
    discritize_labels = []
    surv_labels = []
    for i in range(len(buckets)):
        discritize_labels.append((time_array[:,0]>buckets[i]).float())
        if i == len(buckets) - 1:
            surv_labels.append((time_array[:,0]>buckets[i]).float()*time_array[:,1])
        else:
            temp_array = ((time_array[:,0]>buckets[i]).float()*(time_array[:,0]<buckets[i+1]).float())*time_array[:,1]
            surv_labels.append(temp_array)
    discritize_labels = torch.stack(discritize_labels, dim = 1)
    surv_labels = torch.stack(surv_labels, dim = 1)
    
    return discritize_labels, surv_labels, len(buckets)
