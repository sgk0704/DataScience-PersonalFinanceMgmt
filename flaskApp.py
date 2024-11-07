from flask import Flask, render_template, request
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from PIL import Image
import squarify
import matplotlib.patches as mpatches

app = Flask(__name__)


def get_reccomendation_suggestion(result):
    suggestions = {
        1: {
            'label': "Invest in stocks",
            'suggestion': "Consider researching and investing in stocks that align with your financial goals and risk tolerance. It's important to diversify your investments for long-term growth."
        },
        2: {
            'label': "Save more",
            'suggestion': "Focus on increasing your savings rate by reviewing your expenses and identifying areas where you can cut back. Setting up automatic transfers to a savings account can help you build a financial cushion."
        },
        3: {
            'label': "Reduce expenses",
            'suggestion': "Evaluate your current expenses and identify areas where you can cut back. Consider budgeting and tracking your spending to prioritize essential expenses and reduce discretionary spending."
        },
        4: {
            'label': "Diversify Investments",
            'suggestion': "Explore diversifying your investment portfolio across different asset classes (stocks, bonds, real estate, etc.) to reduce risk and potentially enhance returns over time."
        }
    }
    
    if result in suggestions:
        return suggestions[result]
    else:
        return None
    
def get_health_suggestion(result):
    suggestions = {
        "Very Healthy": {
            'label': "Very Healthy",
            'suggestion': "Maintain your healthy lifestyle with regular exercise, balanced diet, and preventive health check-ups. Continue prioritizing your physical and mental well-being."
        },
        "Healthy": {
            'label': "Healthy",
            'suggestion': "Continue your healthy habits and consider small improvements such as adding more variety to your diet or increasing physical activity. Regular health check-ups are recommended."
        },
        "Fairly Healthy": {
            'label': "Fairly Healthy",
            'suggestion': "Focus on improving certain aspects of your health. Consider adjusting your diet, increasing physical activity, and managing stress levels for better overall health."
        },
        "Average": {
            'label': "Average",
            'suggestion': "Take proactive steps to improve your health. Start with small changes such as incorporating more fruits and vegetables into your diet, exercising regularly, and getting enough sleep."
        },
        "Unhealthy": {
            'label': "Unhealthy",
            'suggestion': "Address immediate health concerns by consulting with a healthcare professional. Focus on adopting healthier habits and seeking medical advice for a personalized health plan."
        },
        "Very Unhealthy": {
            'label': "Very Unhealthy",
            'suggestion': "Seek immediate medical attention and professional guidance to address serious health issues. Make significant lifestyle changes under medical supervision."
        }
    }
    
    return suggestions.get(result, None)

def expense_prediction_plot(form_data,result):
    image_base64 = []
    expense_attributes = [
        'Monthly exp on transport',
        'Monthly exp on food',
        'Monthly exp on education',
        'Recurring monthly payments(Subscriptions, plans ,recharge etc.)',
        'Debt',
        'Monthly savings',
        'Expenses on fresh groceries and whole foods.',
        'expenses on online takeout',
        'expenditure on medicine',
        'expenses on outings',
        'tax payments'
    ]
    expenses = {attr: form_data[attr][0] for attr in expense_attributes}

    expense_values = list(expenses.values())

    # Create the bar graph
    plt.figure(figsize=(12, 8))
    bars = plt.bar(expense_attributes, expense_values, color=plt.cm.tab20.colors)  # Using a color map for diverse colors

    # Adding titles and labels
    plt.title('Monthly Expenses', fontsize=16)
    plt.xlabel('Expense Categories', fontsize=14)
    plt.ylabel('Amount (in currency)', fontsize=14)

    # Rotating the x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Optional: Adding value labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, round(yval, 2), ha='center', fontsize=10)

    # Display the graph
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png',bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    image_base64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))



    predicted_expenditure = result
    df = pd.DataFrame(list(expenses.items()), columns=['Attribute', 'Expense'])
    # Plotting
    plt.figure(figsize=(10, 8))
    
    ax = sns.barplot(x='Expense', y='Attribute', data=df, color='skyblue', label='Actual Expense')
    ax.axvline(x=predicted_expenditure, color='red', linestyle='--', label=f'Predicted Monthly Expenditure: {predicted_expenditure}')
    ax.set_xlabel('Expense Amount')
    ax.set_ylabel('Attribute')
    ax.set_title('Expenses in Different Attributes vs Predicted Monthly Expenditure')
    ax.legend()
    # ax.set_facecolor('black')

    buffer = BytesIO()
    plt.savefig(buffer, format='png',bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    image_base64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    print(type(image_base64))

    print(list(expenses.values()))

    expense = list(expenses.values())
    res = int(result)
    print(res)

    percentages = [value / res * 100 for value in expense]
    labels = [f'{value:.1f}% (${expense})' for key, value, expense in zip(expenses.keys(), percentages, expense)]

    # Colors for tree map rectangles
    colors = plt.cm.Paired(np.linspace(0, 1, len(expense)))

    # Plotting tree map
    plt.figure(figsize=(10, 8))
    squarify.plot(sizes=expense, color=colors, alpha=0.7,label=labels)

    # Add title and turn off axis
    plt.title('Expenses Tree Map')
    plt.axis('off')

    # Create legend
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in zip(expenses.keys(), colors)]
    plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1))
    buffer = BytesIO()
    plt.savefig(buffer, format='png',bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    image_base64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))


    image_path = 'Expense_prediction1.png'  # Replace with your image file path
    image = Image.open(image_path)
    buffer = BytesIO()
    image.save(buffer,format='png')
    buffer.seek(0)
    image_base64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    return image_base64


def monthly_savings_plot(form_data,result):
    image_base64 = []
    expense_attributes = [
        'Monthly exp on transport',
        'Monthly exp on food',
        'Monthly exp on education',
        'Recurring monthly payments(Subscriptions, plans ,recharge etc.)',
        'Debt',
        'Monthly savings',
        'Expenses on fresh groceries and whole foods.',
        'expenses on online takeout',
        'expenditure on medicine',
        'expenses on outings',
        'tax payments'
    ]
    expenses = {attr: form_data[attr][0] for attr in expense_attributes}
    predicted_savings = result
    df = pd.DataFrame(list(expenses.items()), columns=['Attribute', 'Expense'])
    # Plotting
    plt.figure(figsize=(14, 10))
    sns.barplot(x='Expense', y='Attribute', data=df, palette='Blues_d')
    plt.axvline(x=predicted_savings, color='red', linestyle='--', label=f'Predicted Monthly Savings: {predicted_savings}')
    plt.xlabel('Expense Amount')
    plt.ylabel('Attribute')
    plt.title('User Inputs vs Predicted Monthly Savings')
    plt.legend()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    image_base64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))

    image_path = 'Savings1.png'  # Replace with your image file path
    image = Image.open(image_path)
    buffer = BytesIO()
    image.save(buffer,format='png')
    buffer.seek(0)
    image_base64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    return image_base64

def financial_health_plot():
    image_base64 = []
    image_path = 'Financial_health1.png'  # Replace with your image file path
    image = Image.open(image_path)
    buffer = BytesIO()
    image.save(buffer,format='png')
    buffer.seek(0)
    image_base64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))

    image_path = 'Financial_health2.png'  # Replace with your image file path
    image = Image.open(image_path)
    buffer = BytesIO()
    image.save(buffer,format='png')
    buffer.seek(0)
    image_base64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))

    return image_base64


def general():
    df = pd.read_csv('RandData1.csv')
    fig = go.Figure(data=[go.Scatter3d(
        x=df['Monthly pocketmoney'],
        y=df['Monthly exp on food'],
        z=df['No of dependents'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['Monthly pocketmoney'],  # Color by Monthly Income
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    fig.update_layout(
        title='3D Scatter Plot: Monthly Income vs. Debt vs. Monthly Savings',
        scene=dict(
            xaxis_title='Monthly Income',
            yaxis_title='Debt',
            zaxis_title='No of dependents'
        )
    )
    # Convert plot to HTML string
    plot_div = fig.to_html(full_html=False)
    return plot_div


def recommend_plots(result):
    image_base64 = []
    image_path = 'Reccommend.png'  # Replace with your image file path
    image = Image.open(image_path)
    buffer = BytesIO()
    image.save(buffer,format='png')
    buffer.seek(0)
    image_base64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))

    recommend_results = np.array([1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 3, 1, 3, 4, 1, 1, 4, 1, 3, 1, 1, 2, 1, 4, 3, 3, 1, 3, 1, 1, 1, 4, 4, 2, 1, 1, 1], dtype=int)

    # Count occurrences of each value
    unique, counts = np.unique(recommend_results, return_counts=True)
    sizes = counts



    # Recommendation mapping
    recommendation_mapping = {
        1: "Invest in stocks",
        2: "Save more",
        3: "Reduce expenses",
        4: "Diversify investments"
    }

    # Labels
    labels = [recommendation_mapping[i] for i in unique]

    # New prediction data
    #new_prediction = result.tolist()  # Example prediction output
    print("Hello")
    print(result)
    print(type(result))
    new_prediction = [result]
    print(type(new_prediction))
    new_recommendation = [recommendation_mapping[pred] for pred in new_prediction]
    explode = [0.2 if recommendation_mapping[unique[i]] in new_recommendation else 0 for i in range(len(unique))]

    labels = [recommendation_mapping[i] for i in unique]
    # Colors
    colors = plt.cm.Paired(range(len(unique)))
    # colors = ['#ff9999' if recommendation_mapping[unique[i]] in new_recommendation else plt.cm.Paired(i) for i in range(len(unique))]
    # Create a pie chart
    plt.figure(figsize=(5, 5))
    # plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, shadow=True, explode=explode)
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, shadow=True, explode=explode)

    # Add red outline to the exploded segment
    for i, wedge in enumerate(wedges):
        if explode[i] > 0:
            wedge.set_edgecolor('red')
            wedge.set_linewidth(2)
    plt.title('Visualisation of the prediction result')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    image_base64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))

    return image_base64

def health_recommendation_plots(health_prediction,color,range_value,pred_result):
    img_base64 = []
    print(type(int(pred_result)))
    val = int(pred_result)
    fig = go.Figure(go.Indicator(
    mode="gauge",
    value=val,
    title={'text': "Health Prediction"},
    delta={'reference': 0},
    gauge={
        'axis': {'range': [-40, 10]},  # Full range of the gauge
        'steps': [
            {'range': [-40, -30], 'color': "red"},
            {'range': [-30, -20], 'color': "orange"},
            {'range': [-20, -10], 'color': "yellow"},
            {'range': [-10, 0], 'color': "lightgreen"},
            {'range': [0, 7], 'color': "green"},
            {'range': [7, 10], 'color': "darkgreen"}
        ],
        'bar': {'color': "blue"},
        'threshold': {
            'line': {'color': color, 'width': 4},
            'thickness': 0.75,
            'value': val
            }
        }
    ))

    # Update layout to display the health prediction
    fig.update_layout(
        annotations=[go.layout.Annotation(
            x=0.5,
            y=0.1,
            text=f"<b>{health_prediction}</b>",
            showarrow=False,
            font=dict(size=20)
        )]
    )

    img_bytes = fig.to_image(format="png")
    file_path = 'output_image.png'

    # Save the image bytes to a file
    with open(file_path, 'wb') as f:
        f.write(img_bytes)
    buffer = BytesIO(img_bytes)
    buffer.seek(0)

    # Encode the image to base64
    img_base64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    return img_base64
    


def expense_prediction(input_data):
    model = joblib.load('Expense_prediction.pkl')
    pred_result = model.predict(input_data)
    return pred_result[0]

def monthly_savings(input_data):
    model = joblib.load('Savings.pkl')
    pred_result = model.predict(input_data)
    return pred_result[0]

def financial_health(input_data):
    model = joblib.load('Financial_health.pkl')
    pred_result = model.predict(input_data)
    if pred_result < 0:
        financial_health_result = "Poor Financial Health"
    elif pred_result > 0 and pred_result < 1000:
        financial_health_result = "Good Financial Health"
    else:
        financial_health_result = "Very Good Financial Health"
    return financial_health_result

def recommend(input_data):
    model = joblib.load('Reccommend.pkl')
    pred_result = model.predict(input_data)
    return pred_result[0]

def health_recommendation(input_data):
    filtered_data_input = {
        'Age': input_data['Age'],
        'how often do u dine out?': input_data['how often do u dine out?'],
        'expenditure on medicine': input_data['expenditure on medicine'],
        'what types of food do u spen money on': input_data['what types of food do u spen money on'],
        'Gender': input_data['Gender']
    }
    health_data = pd.DataFrame(filtered_data_input)
    health_data_array = health_data.to_numpy()
    model = joblib.load('Health.pkl')
    pred_result = model.predict(health_data_array)
    if pred_result >= 7:
        health_prediction = "Very Healthy"
        color = "darkgreen"
        range_value = [7, 10]
    elif pred_result >= 0:
        health_prediction = "Healthy"
        color = "green"
        range_value = [0, 7]
    elif pred_result >= -10:
        health_prediction = "Fairly Healthy"
        color = "lightgreen"
        range_value = [-10, 0]
    elif pred_result >= -20:
        health_prediction = "Average"
        color = "yellow"
        range_value = [-20, -10]
    elif pred_result >= -30:
        health_prediction = "Unhealthy"
        color = "orange"
        range_value = [-30, -20]
    else:
        health_prediction = "Very Unhealthy"
        color = "red"
        range_value = [-40, -30]
    return health_prediction,color,range_value,pred_result

@app.route('/form')
def landing():
    return render_template('form1.html')

@app.route('/')
def index():
    try:
        with open('number.txt', 'r') as file:
            i = int(file.read().strip())
    except FileNotFoundError:
        i = 0
    i = i + 1
    with open('number.txt', 'w') as file:
        file.write(str(i))
    return render_template('index.html',result = i)

@app.route('/predict', methods=['POST'])
def predict():
    form_data = {
        'Age': [int(request.form['age'])],
        'Gender': [int(request.form['gender'])],
        'Place of Stay': [int(request.form['place_of_stay'])],
        'Relationship status': [int(request.form['relationship_status'])],
        'No of dependents': [int(request.form['no_of_dependents'])],
        'Monthly pocketmoney': [int(request.form['monthly_pocketmoney'])],
        'Monthly exp on transport': [int(request.form['monthly_exp_on_transport'])],
        'Monthly exp on food': [int(request.form['monthly_exp_on_food'])],
        'Monthly exp on education': [int(request.form['monthly_exp_on_education'])],
        'notable investments': [int(request.form['notable_investments'])],
        'Recurring monthly payments(Subscriptions, plans ,recharge etc.)': [int(request.form['recurring_monthly_payments'])],
        'Debt': [int(request.form['debt'])],
        'Monthly savings': [int(request.form['monthly_savings'])],
        'Financial goals': [int(request.form['financial_goals'])],
        'Do you follow a monthly budget ?': [int(request.form['monthly_budget'])],
        'how often do u dine out?': [int(request.form['dine_out'])],
        'how often do you make impulse purchases?': [int(request.form['impulse_purchases'])],
        'do you currently invest in any financial products like stocks,mutual funds or cryptocurrencies': [int(request.form['financial_products'])],
        'how comfortable are you with taking financial risk in your investments': [int(request.form['financial_risk'])],
        'Do you use any mobile apps or online platforms to manage your finances and investments': [int(request.form['finance_apps'])],
        'Expenses on fresh groceries and whole foods.': [int(request.form['groceries'])],
        'expenses on online takeout': [int(request.form['online_takeout'])],
        'expenditure on medicine': [int(request.form['medicine'])],
        'expenses on outings': [int(request.form['outings'])],
        'tax payments': [int(request.form['tax_payments'])],
        'what types of food do u spen money on': [int(request.form['food_types'])]
    }

    input_data = pd.DataFrame(form_data)
    if 'button1' in request.form:
        result = expense_prediction(input_data)
        plots = expense_prediction_plot(form_data,result)
        name = "Monthly Expenditure"
    elif 'button2' in request.form:
        result = monthly_savings(input_data)
        plots = monthly_savings_plot(form_data,result)
        name = "Monthly Savings"

    elif 'button3' in request.form:
        result = financial_health(input_data)
        plots = financial_health_plot()
        name = "Financial Health"
    #savings_result = monthly_savings(input_data)

    elif 'button4' in request.form:
        result1 = recommend(input_data)
        print(type(result1))
        result = get_reccomendation_suggestion(result1)
        plots = recommend_plots(result1)
        #Invest in stocks = 1
        #Save more = 2 
        #Reduce expenses = 3 
        # Diversify investments = 4
        name = "Recommendation"
    
    elif 'button5' in request.form:
        health_result,color,range_value,pred_result = health_recommendation(form_data)
        print(health_result,color,range_value,pred_result)
        result = get_health_suggestion(health_result)
        plots = health_recommendation_plots(health_result,color,range_value,pred_result)
        name = "your health"
    
    elif 'button6' in request.form:
        result = "none"
        plots = general()
        name = "None"
    
    return render_template('result.html',result =[result, plots,name])

if __name__ == '__main__':
    app.run(debug=True)
