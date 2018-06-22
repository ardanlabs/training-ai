# Introduction to productionizing ML/AI

This material introduces some common pain points and pitfalls that people fall into when trying to productionize data science work. Once your are done with this material, you will understand what the common pain points are and the guiding principles that will help us overcome them.

Much of this section of the course is meant to be a discussion. That discussion will be centered around an example AI workflow in a Jupyter notebook. This guide will walk you through:

1. [Running the example python workflow](#1-running-the-example-python-workflow)
2. [Discussing how we might productionize the workflow](#2-discussing-how-we-might-productionize-the-workflow)
3. [Production/deployed ML/AI workflows](#3-productiondeployed-mlai-workflows)

It also includes a [list of resources](#resources) for those that want to dive in a little bit deeper.

## 1. Running the example Python workflow

While you have Jupyter up and running, click on the `example_sklearn_workflow.ipynb` notebook (which you should see in the `productionizing` directory). This will bring up our example notebook. We will run through and discuss this notebook interactively in class. As a reminder, our example problem for the day will be the [Iris flower classification problem](https://en.wikipedia.org/wiki/Iris_flower_data_set), and we are using scikit-learn as a jumping off point.

## 2. Discussing how we might productionize the workflow

It might not be clear what "productionize" means or how we might try doing that. This will be discussed in class, but before or during our conversation think about:

- What characteristics should a production AI workflow exhibit?
- What differentiates a production AI workflow from a non-production workflow?
- In what environments do production AI workflows run? 
- Are scaling and deployment the same thing?
- How might data change as we move to production?
- How do we want to manage our production AI workflows?
- Do we treat all parts of our workflow the same as we move them to production?
- What happens when our workflows fail at scale?
- How does data ingress/egress happen at scale?
- How will our workflow interact with other pieces of infrastruction or other applications?
- How do we handle our dependencies?

By the end of this discussion, we should decide on set of guidelines that should drive how we productionize our Python workflow.

## 3. Production/deployed ML/AI workflows

Now that we know the guidelines that we should follow as we work towards a production ML/AI deployment, let's review some common patterns. This will again be focused around diagrams drawn in class and the workshop slides.

## Resources

Technical resources:

- [Jupyter](http://jupyter.org/)
- [Pandas](https://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Introduction to Neural Networks](http://blog.kaggle.com/2017/11/27/introduction-to-neural-networks/)

Articles:

- [A Guide to Building a High Functioning Data Science Department](http://multithreaded.stitchfix.com/blog/2016/03/16/engineers-shouldnt-write-etl/)  
- [Data Science at the Speed of Hype](http://www.john-foreman.com/blog/surviving-data-science-at-the-speed-of-hype)   
- [How we do Data Science at People Pattern](https://www.peoplepattern.com/post.html#!/how-we-do-data-science-at-people-pattern)  
- [Doing Data Science at Twitter](https://medium.com/@rchang/my-two-year-journey-as-a-data-scientist-at-twitter-f0c13298aee6)
- [Data Science Bill of Rights](http://www.pachyderm.io/dsbor.html)

___
All material is licensed under the [Apache License Version 2.0, January 2004](http://www.apache.org/licenses/LICENSE-2.0).
