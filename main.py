from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

data = Dataset.load_builtin('ml-100k')

trainset, testset = train_test_split(data, test_size=0.25)

algo = SVD()

algo.fit(trainset)

predictions = algo.test(testset)

rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')

user_id = str(1)  
user_predictions = []

for movie_id in range(1, 1683):  
    pred = algo.predict(user_id, str(movie_id))
    user_predictions.append((pred.iid, pred.est))

user_predictions.sort(key=lambda x: x[1], reverse=True)

top_n = 10
top_recommendations = user_predictions[:top_n]
print(f'\nTop {top_n} Recommendations for User {user_id}:')
for movie_id, estimated_rating in top_recommendations:
    print(f'Movie ID: {movie_id}, Estimated Rating: {estimated_rating}')
