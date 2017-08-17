from flask import render_template
from hanna5descher import app
from flask import request
from hanna5descher.modelRF_SMOTE import modelRF
import pandas as pd
import datetime as dt

@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/skills')
def skills():
    return render_template('skills.html')

@app.route('/publications')
def publications():
    return render_template('publications.html')

@app.route('/projects')
def projects():
    return render_template('projects.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/dashboard')
def dashboard():
    defaultTerm = 27 # the most current term completed (data received on 7/20/2017, termid = 28 in progress)
    nUsersDisplay = 50 # number of users displayed in the table

    inputTerm = request.args.get('term')
    if inputTerm is not None and inputTerm != 'select':
        currentTerm = int(inputTerm[-2:])
    else:
        currentTerm = defaultTerm

    # Daily user product usage per week (weekdays)
    filename_usage = 'weekly_usage.csv'
    dfUsage = pd.read_csv('hanna5descher/database/' + filename_usage)
    dfUsage = dfUsage.loc[dfUsage['termid'] == currentTerm]
    usagedata = []
    for index, day in dfUsage.iterrows():
        usagedata.append(dict(label=day['day_str'], count=int(day['percentage'])))

    # stateID to state name (index = stateID)
    filename_state = 'stateid_chart.csv'
    dfState = pd.read_csv('hanna5descher/database/' + filename_state, index_col=0)

    # Read in features per term
    filename_features = 'features_perTerm_fillNA1_timeLag7_timestep2T1.csv'
    readFeatures = pd.read_csv('hanna5descher/database/'+filename_features, index_col = 0)
    dfFeatures = readFeatures.loc[readFeatures['termid'] == currentTerm]
    #print('Number of users:', len(dfFeatures))

    # Run model
    dfModel = modelRF(dfFeatures)

    # Create model outputs to html: transform dataframe to dictionary
    userdata = []
    topusers = 1
    for index, user in dfModel.iterrows():
        if topusers <= nUsersDisplay:
            userdata.append(dict(userid = int(user['userid']),
                                start_service = int(user['start_service']),
                                location = dfState.loc[user['location']].iloc[0],
                                churn_prob = float("{0:.2f}".format(user['churn_prob'])),
                                risk = user['risk']))
            topusers += 1

    return render_template("dashboard.html",
                           termid = currentTerm,
                           userdata = userdata,
                           usagedata = usagedata)
