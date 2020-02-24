import packages_to_import as p
#from understanding_data import understanding_data
import cleaning_and_preparing_train_data
import training
from training import *
import time

start = time.time()
# data=p.pd.read_csv("Reviews.csv")


#understanding_data(data)
# cleaning_and_preparing_train_data.df(data)
text_query=input("Enter the review : ")
test_data = p.pd.DataFrame(p.np.column_stack([text_query]),columns=['Text'])
tt=train_test(test_data)
# tt.train()

tt.testing()
predicted=tt.testing()
predicted="Positive" if predicted > 0.5 else "Negative"
print("The review is",predicted)

end = time.time()
print(f"Runtime of the program is {end - start}")
