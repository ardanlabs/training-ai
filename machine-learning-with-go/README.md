![Alt text](https://docs.google.com/drawings/d/e/2PACX-1vT37glyZXd8ViXedt0LCSpzsbWCUSSLhWuR3o5_74tL92fh7zeIo3hVtCzhnpw8IeWAM-KcI419cIkm/pub?w=745&h=310)

# Machine Learning with Go

This is material for any Go developer, data scientist, analyst, or statistician who wishes to learn how to build robust machine learning (ML) applications in Go. This class provides an intensive, comprehensive and idiomatic view on training, utilizing, evaluating, and deploying machine learning models using Go. We believe this class is perfect for anyone wishing to build data-driven applications that produce valuable insights, have reproducible behavior, and can be deployed within modern architectures.

- [Slides from the class](https://docs.google.com/presentation/d/1BMRPCNPptXsLxw40-1c7HG2UEXOK-sBp8OfloJrCv6I/edit?usp=sharing)
- Instructors 
  - Daniel Whitenack - author of [Machine Learning with Go](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-go), [website/blog](http://www.datadan.io/), [online courses](http://learn.datadan.io/), [twitter](https://twitter.com/dwhitena), [github](https://github.com/dwhitena)
  - Diana Ortega - [twitter](https://twitter.com/dicaormu)
- Prerequisties/getting started:
  - You will need to ssh into a cloud instance. Remind yourself of how to do that and install a client if needed:
    - On a Mac or Linux machine, you should be able to ssh from a terminal (see these [Mac instructions](http://accc.uic.edu/answer/how-do-i-use-ssh-and-sftp-mac-os-x) and [Linux instructions](https://www.digitalocean.com/community/tutorials/how-to-use-ssh-to-connect-to-a-remote-server-in-ubuntu)).
    - On a Windows machine, you can either [install and use an ssh client (I recommend PuTTY)](https://www.putty.org/) or [use the WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10).
  - You will also need to work a bit at the command line. If you are new to the command line or need a refresher, look through [this quick tutorial](https://lifehacker.com/5633909/who-needs-a-mouse-learn-to-use-the-command-line-for-almost-anything).
- If you need further help productionizing Go ML/AI workflows, want to bring this class to your company, or just have ML/AI related questions, [Ardan Labs](https://www.ardanlabs.com/) is here to help! Reach out to the instructor using the links above or via the [Ardan Labs website](https://www.ardanlabs.com/). 

*Note: This material has been designed to be taught in a classroom environment. The code is well commented but missing some of the contextual concepts and ideas that will be covered in class.*

## ML and the ML workflow

This material introduces the basics of machine learning and the workflow that should be used when developing and utilizing machine learning models. Once you are done with this material, you will be exposed to the most commonly used machine learning techniques and you will understand the significance of data profiling, training, evaluation, validation, and inference. 

[ML and the ML workflow](ml_intro)

## ML with Go

This material introduces some Go packages and frameworks that will help us implement ML in Go. Once you are done with this material, you will know where to look to find ML-related packages for Go, have some hands-on experience working with these packages, and understand the utility of Go for ML.

[ML with Go](ml_with_go)

## Building a complete Go-based ML workflow

This material walks you through a lab in which you will implement a full ML workflow with Go, from data ingress to training to evaluation to inference. Once you are done with this material, you know how to implement and deploy the stages of the ML workflow in Go (for at least one type of ML model and data), and you will be able to transfer this workflow scaffolding to other problems.

[Building a complete Go-based ML workflow](ml_workflow)

___
All material is licensed under the [Apache License Version 2.0, January 2004](http://www.apache.org/licenses/LICENSE-2.0).
