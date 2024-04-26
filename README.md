# ImprovEYESCode
The leading cause of blindness in working-age adults is diabetic retinopathy: a complication of diabetes that causes vision loss due to damage in the capillaries. Diabetic retinopathy is predicted to impact 18 million people by 2030. Stanford Medicine says, “There is no cure for diabetic retinopathy. But treatment works very well to prevent vision loss. The sooner the condition is found, the easier it is to prevent.” However, due to diabetic retinopathy’s asymptomatic progression, the illness isn’t typically found until it reaches its final stage, the point at which treatment is no longer an option. By the time diabetic retinopathy is diagnosed, vision loss is permanent and untreatable. Introducing a computational dimension to the problem, the objective of this project was to develop a supervised machine learning algorithm to detect diabetic retinopathy before it reaches its untreatable stages. The NumPy, Pandas, and TensorFlow libraries were used to achieve this within a convolutional neural network. Utilizing EfficientNetB3 as the base layer, Adamax as the optimizer, and categorical crossentropy loss for the cost function, a robust softmax regression pipeline with over 900 thousand changeable weights, named ImprovEYES, was created. ImprovEYES drew connections between each of the phases of retinopathy and created “rules” for future diagnosis.
It segregated images with microaneurysms and intraretinal hemorrhages in the mild stage, and classified pictures with cotton wool spots and hard exudates in the moderate stage. ImprovEYES grouped together photos with vitreous hemorrhages, new blood vessels, and fibrous tissue in the proliferative stage, and classified pictures with macular edema, detached retina, and fibrous tissue in the severe stage. When applied to the previously unseen test dataset, ImprovEYES reached an accuracy of a staggering 94.86%, showing promising potential for real-world applications. 
