# Task Process 
## Job 1: 
         Job 1 will keep an eye on the GitHub repository as soon as the developer pushes something. This job will automatically copy everything in the folder of my base os. (I am using RHEL 8 as my base os (VM)).

## Job 2:
     Success of Job 1 will trigger job 2. This job will launch a docker container from the image which we created from Dockerfile.

## Job 3:
        After successfully launching the Container, Jenkins will trigger this job. This job will execute the file which pushed by the developer and has the main code to train the model. (mycode.py)

## Job 4: 
        This job is to check the accuracy of our executed code if it satisfies the requirement this will notify(email) developer else will run the next job for some tweaking.

## Job 5:   
       If mycode.py runs successfully but give less accuracy than what developer desire then Jenkins should automatically tweak something and by various hit and trials will try to increase the accuracy. In order to achieve this thing developer will push one more file along with mycode.py that is tweak.py. This will help Jenkins to take tests and build the model again and again till the desired accuracy is achieved.

## Job 6: 
      This job is to check the accuracy of our tweaked code if it satisfies the requirement this will notify(email) developer else will run the previous job (job 5) again for some tweaking, this loop will continue till we get desired accuracy.

## Job 7:
       This job will be a monitoring job. It will keep an eye on running container. If it found the container crashed, it will immediately launch a new container with the same configuration.

# Let’s Started
## Step-1  Create Dockerfile with tensorflow and keras
  In RHEL8 first make a directory that will store all the data or the program for our machine learning model.
   ### mkdir mlops
Now the jenkins will automatically copy the files in this folder.
## Dockerfile
### FROM centos
### RUN RUN yum install python36    -y
### RUN  pip3 install  --upgrade pip
### RUN  pip3 install  tensorflow
### RUN  pip3 install  keras
### CMD [“python3”,“/mlops/mycode.py”]

## Job-1 Pull Github Code
  When the developer will commit any code to GitHub , this job will copy that code into the local repository in our system. For this I have used remote trigger and add to post-commit file to keep on checking the remote repository for any changes.
  In jenkins, we 
  ##### Source Code Management: provides url of github repository
  ##### Execute shell : sudo cp -rvf * /mlops/mycode.py
  When job -1 is executed by jenkins .It copy the file in mlops dirrectory.
## Job-2 Check code and launch CNN environment
  The checkcode.py file is used to check that code belongs to CNN or not.IF the code is belongs to CNN ,then Job-2 launch an OS CNNos by the Dockerfile.
  ##### Build Triggers:Build after other projects are built
  ##### Execute Shell:
  if [[ "$(sudo python3 /mlops/checkcode.py)" == "CNN"
then
if sudo docker ps -a  | grep CNNos
then
sudo docker rm CNNos
fi
sudo docker run --name CNNos -v /root/mlops:/root/mlops1 mlops:v1
else
echo" The image is not found"
fi

Here we mount mlops directory to mlops1 .And launch CNN environment by mlops:v1 image
 ## Job-3 : 
 Job 2 will check the code, launch the respective container, train the model and copy the accuracy to file accuracy.txt
In the mycode.py code, we are using command line argument to tune the model.After successfully launching the Container, Jenkins will trigger this job. This job will execute the file which pushed by the developer and has the main code to train the model. (mycode.py)
## Job-4 : Predict Accuracy
This is used to check accuracy of the trained model
accuracy.txt file is used to check accuracy .In jenkin job we write in
##### Execute shell:
MIN=80
ACCURACY='sudo cat /root/mlops1/accuracy.txt'
ACCURACY=${ACCURACY%.*}

if [ $ACCURACY -1t $MIN ]
then
 echo "Required accuracy not achieved"
 exit 0
else
 echo"Again train model"
 fi
 
 ## Job-5 Analyse Accuracy and change by tweak.py file
 Checks accuracy , if accuracy is less than required , then tweak the code using program tweaker.py and again start job 2 i.e. see code and launch to start the container and run the model once again.2)If accuracy requirement is met , call job 5 i.e. model create success

Now lets see how tweaker.py tweaks the code...

When tweaker.py is called , it compares the accuracies old (initially 0) and new (gained from running the container) . If the accuracy has increased then it increases the value of the first hyper parameter(here number of filters) of the base convolve layer.

Also it changes the initial 0 accuracy to new accuracy received in the data.txt file for next build calculations.

As soon as the hyper parameter value is changed , the job2 is re run to see the accuracy.

Now , if the accuracy would have increased , it means that the value increased was good and can be increased further , so it increases that parameter's value further.

But if it finds that the accuracy has decreased , then our program tweaker.py changes the parameter's value to its initial value and now starts changing the value of the next hyper parameter (which is in our case is filter size).

In every call , it repeats this process until in that layer , no more hyper parameters can be increased and when such a case arises , it goes on to add another layer and do all the above processes once again in the new layer.
## Job 6 :Send mail to developer stating the model accuracy
status and copy the model.h5(final model) to /root/mlops/ by jenkins
This is triggered when the required accuracy is met and the input file is mailed to the developer to help the developer know the correct value of the hyper parameters.
## Job 7 : Job 7 will keep on monitoring the KNNos and relaunch the container if it stops.

