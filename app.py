from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load ML model and encoders
model = joblib.load('risk_model.pkl')
gender_encoder = joblib.load('gender_encoder.pkl')
risk_encoder = joblib.load('risk_encoder.pkl')

# --- BMI Calculator ---
def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    bmi = weight / (height_m ** 2)
    return round(bmi, 2)

# --- BMI Classification Logic ---
def classify_bmi(bmi):
    if bmi < 18.5:
        return "Underweight", "Gain weight with high-protein meals."
    elif 18.5 <= bmi < 24.9:
        return "Normal", "Maintain your lifestyle with a balanced diet."
    elif 25 <= bmi < 29.9:
        return "Overweight", "Reduce sugar intake and exercise regularly."
    else:
        return "Obese", "Consider medical advice and a calorie-deficit diet."

# --- Predict Risk from Model ---
def predict_risk(age, gender, bmi):
    gender_encoded = gender_encoder.transform([gender])[0]
    features = [[age, gender_encoded, bmi]]
    pred = model.predict(features)[0]
    return risk_encoder.inverse_transform([pred])[0]

# --- Weekly Diet Plan Based on Age, Gender, Preference ---
def get_weekly_diet(age, gender, preference):
    if age < 30:
        group = "18-30"
    elif 30 <= age <= 50:
        group = "30-50"
    else:
        group = "50-70"

    diet = {
        "18-30": {
            "Male": {
                "Veg": {
                    "Monday": ["Oats + Banana", "Dal + Brown Rice", "Paneer Bhurji + Roti"],
                    "Tuesday": ["Upma + Almonds", "Rajma + Rice", "Tofu Curry + Roti"],
                    "Wednesday": ["Idli + Sambar", "Vegetable Pulao", "Palak Paneer"],
                    "Thursday": ["Poha + Nuts", "Veg Wraps + Yogurt", "Chickpea Stew"],
                    "Friday": ["Smoothie + Toast", "Mixed Lentil Rice", "Stuffed Paratha + Curd"],
                    "Saturday": ["Paratha + Butter", "Khichdi + Salad", "Paneer Tikka"],
                    "Sunday": ["Cornflakes + Milk", "Soya Pulao", "Mixed Veg Curry"]
                },
                "Non-Veg": {
                    "Monday": ["Boiled Eggs + Toast", "Grilled Chicken + Quinoa", "Fish Curry + Rice"],
                    "Tuesday": ["Omelette + Milk", "Chicken Biryani", "Tuna Salad"],
                    "Wednesday": ["Smoothie + Eggs", "Chicken Wrap", "Grilled Fish"],
                    "Thursday": ["Scrambled Eggs", "Egg Curry + Rice", "Chicken Stir Fry"],
                    "Friday": ["Egg Bhurji + Toast", "Roast Chicken", "Boiled Egg Salad"],
                    "Saturday": ["Chicken Sandwich", "Meat Stew + Rice", "Grilled Fish + Broccoli"],
                    "Sunday": ["Protein Shake", "Chicken Pulao", "Fish Cutlet"]
                }
            },
            "Female": {
                "Veg": {
                    "Monday": ["Oats + Berries", "Dal Khichdi", "Paneer Curry + Chapati"],
                    "Tuesday": ["Dalia + Curd", "Veg Pulao", "Lentil Soup + Roti"],
                    "Wednesday": ["Idli + Chutney", "Veg Paratha", "Soya Curry"],
                    "Thursday": ["Fruit Smoothie", "Rajma Wrap", "Vegetable Stir Fry"],
                    "Friday": ["Muesli + Milk", "Chole Rice", "Stuffed Capsicum"],
                    "Saturday": ["Upma + Curd", "Kadhi + Rice", "Tofu Bhurji"],
                    "Sunday": ["Multigrain Toast", "Green Salad + Chapati", "Vegetable Soup"]
                },
                "Non-Veg": {
                    "Monday": ["Egg Whites + Toast", "Grilled Chicken Salad", "Fish Soup"],
                    "Tuesday": ["Omelette + Juice", "Chicken Rice", "Tuna Wrap"],
                    "Wednesday": ["Smoothie + Boiled Egg", "Egg Fried Rice", "Chicken Skewers"],
                    "Thursday": ["Egg Sandwich", "Chicken Wrap", "Fish Tikka"],
                    "Friday": ["Scrambled Eggs", "Chicken Stew", "Grilled Fish"],
                    "Saturday": ["Boiled Egg + Toast", "Chicken Cutlet", "Soup + Bread"],
                    "Sunday": ["Protein Pancake", "Biryani", "Fish Curry"]
                }
            }
        },
        "30-50": {
            "Male": {
                "Veg": {
                    "Monday": ["Oats + Walnuts", "Moong Dal + Rice", "Veg Stew + Roti"],
                    "Tuesday": ["Vegetable Poha", "Tofu Rice Bowl", "Lauki Curry"],
                    "Wednesday": ["Multigrain Toast", "Paneer Bhurji + Roti", "Kadhi + Rice"],
                    "Thursday": ["Cornflakes + Milk", "Sprouts Salad", "Baingan Bharta"],
                    "Friday": ["Smoothie Bowl", "Bhindi + Dal", "Cabbage Sabzi"],
                    "Saturday": ["Idli + Sambar", "Vegetable Upma", "Tofu Stir Fry"],
                    "Sunday": ["Muesli + Curd", "Khichdi + Papad", "Dum Aloo + Chapati"]
                },
                "Non-Veg": {
                    "Monday": ["Boiled Eggs + Tea", "Grilled Chicken", "Fish Curry + Brown Rice"],
                    "Tuesday": ["Scrambled Eggs", "Egg Pulao", "Roast Chicken"],
                    "Wednesday": ["Protein Shake", "Fish Fry", "Egg Curry"],
                    "Thursday": ["Egg Toast", "Chicken Salad", "Fish Stew"],
                    "Friday": ["Egg Bhurji", "Grilled Chicken Wrap", "Chicken Rice"],
                    "Saturday": ["Omelette + Juice", "Tuna Sandwich", "Fish + Veggies"],
                    "Sunday": ["Egg Whites + Chapati", "Chicken Biryani", "Grilled Fish"]
                }
            },
            "Female": {
                "Veg": {
                    "Monday": ["Methi Paratha", "Lauki + Dal", "Paneer + Rice"],
                    "Tuesday": ["Vegetable Dalia", "Khichdi", "Tofu Curry"],
                    "Wednesday": ["Poha + Curd", "Sprouts Salad", "Tinda + Roti"],
                    "Thursday": ["Smoothie", "Rajma + Rice", "Vegetable Soup"],
                    "Friday": ["Muesli", "Veg Wrap", "Baked Veg Cutlet"],
                    "Saturday": ["Curd Rice", "Moong Dal", "Soya Tikka"],
                    "Sunday": ["Cornflakes", "Kadhi + Chapati", "Mixed Veg Curry"]
                },
                "Non-Veg": {
                    "Monday": ["Boiled Eggs", "Grilled Chicken Salad", "Tuna Rice"],
                    "Tuesday": ["Egg Bhurji", "Chicken Curry + Rice", "Fish Cutlet"],
                    "Wednesday": ["Omelette + Toast", "Chicken Stew", "Egg Masala"],
                    "Thursday": ["Scrambled Eggs", "Tuna Wrap", "Fish Curry"],
                    "Friday": ["Egg Salad", "Chicken Wrap", "Soup + Chicken Balls"],
                    "Saturday": ["Egg + Chapati", "Chicken Biryani", "Grilled Fish"],
                    "Sunday": ["Smoothie + Eggs", "Egg Pulao", "Boiled Fish"]
                }
            }
        },
        "50-70": {
            "Male": {
                "Veg": {
                    "Monday": ["Oats + Almonds", "Soft Dal + Rice", "Lauki + Roti"],
                    "Tuesday": ["Dalia + Curd", "Khichdi + Ghee", "Tofu Sabzi"],
                    "Wednesday": ["Poha + Curd", "Mixed Dal", "Tinda + Roti"],
                    "Thursday": ["Upma + Herbal Tea", "Rice + Moong Dal", "Baingan Bharta"],
                    "Friday": ["Cornflakes", "Pumpkin Curry", "Vegetable Stew"],
                    "Saturday": ["Boiled Veg + Roti", "Kadhi + Rice", "Tofu + Chapati"],
                    "Sunday": ["Multigrain Toast", "Lentil Soup", "Light Paneer Curry"]
                },
                "Non-Veg": {
                    "Monday": ["Boiled Eggs", "Grilled Fish", "Chicken Soup"],
                    "Tuesday": ["Omelette", "Chicken Pulao", "Fish Stew"],
                    "Wednesday": ["Egg Sandwich", "Tuna Rice", "Egg Curry"],
                    "Thursday": ["Scrambled Eggs", "Boiled Chicken", "Fish + Rice"],
                    "Friday": ["Egg Whites", "Soup + Bread", "Fish Salad"],
                    "Saturday": ["Smoothie + Boiled Egg", "Chicken Wrap", "Egg Omelette"],
                    "Sunday": ["Protein Shake", "Biryani", "Fish Curry"]
                }
            },
            "Female": {
                "Veg": {
                    "Monday": ["Oats + Milk", "Moong Dal Khichdi", "Tinda + Roti"],
                    "Tuesday": ["Poha", "Dalia + Curd", "Lauki Curry"],
                    "Wednesday": ["Upma + Tea", "Vegetable Stew", "Chapati + Bhindi"],
                    "Thursday": ["Smoothie", "Tofu Wrap", "Pumpkin Soup"],
                    "Friday": ["Muesli + Milk", "Kadhi + Rice", "Spinach + Paneer"],
                    "Saturday": ["Cornflakes", "Rajma + Rice", "Stuffed Paratha"],
                    "Sunday": ["Curd Rice", "Soft Dal", "Mixed Veg Curry"]
                },
                "Non-Veg": {
                    "Monday": ["Boiled Eggs", "Fish Curry", "Chicken Soup"],
                    "Tuesday": ["Egg Bhurji", "Boiled Chicken", "Tuna Salad"],
                    "Wednesday": ["Smoothie + Egg", "Fish Rice", "Grilled Chicken"],
                    "Thursday": ["Omelette", "Chicken Wrap", "Egg Stew"],
                    "Friday": ["Scrambled Egg", "Chicken Sandwich", "Fish Veg Soup"],
                    "Saturday": ["Protein Shake", "Chicken Cutlet", "Fish Fry"],
                    "Sunday": ["Egg Toast", "Chicken Biryani", "Light Fish Curry"]
                }
            }
        }
    }

    return diet.get(group, {}).get(gender, {}).get(preference, {})

# --- Main Route ---
@app.route("/", methods=["GET", "POST"])
def index():
    chart_data = {}
    if request.method == "POST":
        name = request.form["name"]
        age = int(request.form["age"])
        gender = request.form["gender"]
        diet = request.form["diet"]
        weight = float(request.form["weight"])
        height = float(request.form["height"])

        bmi = calculate_bmi(weight, height)
        category, suggestion = classify_bmi(bmi)
        meal_plan = get_weekly_diet(age, gender, diet)
        risk = predict_risk(age, gender, bmi)
        chart_data = {"bmi": bmi, "category": category}

        return render_template(
            "index.html", name=name, bmi=bmi, category=category,
            suggestion=suggestion, meal_plan=meal_plan, chart_data=chart_data,
            diet=diet, show_result=True, risk=risk
        )
    return render_template("index.html", show_result=False)

if __name__ == '__main__':
    app.run(debug=True)
