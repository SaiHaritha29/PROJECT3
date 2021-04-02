import plotly.graph_objects as go
labels = ['Cancer','Chronic and Respiratory_Diseases','Heart Disease','Accidents','Stroke','Aizheimer_Disease','Diabetes','Flu_pneumonia','Kidney_Disease','Suicide']
values = [595930,155041,633842,146571,140323,110561,79535,57062,49959,44193]

# pull is given as a fraction of the pie radius
fig = go.Figure(data=[go.Pie(labels=labels, values=values, title='Population of American continent' ,pull=[0, 0, 0.2, 0])])
fig.show()

