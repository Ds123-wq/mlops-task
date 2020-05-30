Task Process 
Job 1: Job 1 will keep an eye on the GitHub repository as soon as the developer pushes something. This job will automatically copy everything in the folder of my base os. (I am using RHEL 8 as my base os (VM)).
Job 2: Success of Job 1 will trigger job 2. This job will launch a docker container from the image which we created from Dockerfile.
Job 3: After successfully launching the Container, Jenkins will trigger this job. This job will execute the file which pushed by the developer and has the main code to train the model. (mycode.py)
Job 4: This job is to check the accuracy of our executed code if it satisfies the requirement this will notify(email) developer else will run the next job for some tweaking.
Job 5: If mycode.py runs successfully but give less accuracy than what developer desire then Jenkins should automatically tweak something and by various hit and trials will try to increase the accuracy. In order to achieve this thing developer will push one more file along with mycode.py that is tweak.py. This will help Jenkins to take tests and build the model again and again till the desired accuracy is achieved.
Job 6: This job is to check the accuracy of our tweaked code if it satisfies the requirement this will notify(email) developer else will run the previous job (job 5) again for some tweaking, this loop will continue till we get desired accuracy.
Job 7:This job will be a monitoring job. It will keep an eye on running container. If it found the container crashed, it will immediately launch a new container with the same configuration.

