
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from plotly.subplots import make_subplots

pio.renderers.default='browser'

import matplotlib.pyplot as plt
import numpy as np






def scatter3d(fig,data,legend, mode = 'markers',line_dict = {},marker_dict = {}):

    marker_dict = marker_dict if 'markers' in mode else {}
    
    # Include line dict if mode includes 'lines'
    line_params = line_dict if 'lines' in mode else {}
    
      
    fig.add_trace(go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        mode=mode,
        marker=marker_dict,
        line = line_params,
        name = legend
    ))
    
    # Update layout to set aspectmode to 'cube'
    fig.update_layout(scene=dict(
        aspectmode='data'  # Ensures x, y, z axes have the same scale
    ))
    return fig

def plot_cones(fig, points, normals,skip = 10,sizeref = 1000,opacity = 0.5):

    fig.add_trace(go.Cone(
    x=points[::skip,0],
    y=points[::skip,1],
    z=points[::skip,2],
    u=normals[::skip,0],
    v=normals[::skip,1],
    w=normals[::skip,2],
    opacity= opacity,
    sizemode="absolute",
    showscale = False,
    sizeref=sizeref))
    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),aspectmode = 'data',
                             camera_eye=dict(x=1.2, y=1.2, z=0.6)))


def plot_skeleton(bones,fig,marker_dict,line_dict, name = ['neck_head','neck_thorax','thorax_abdomen','right_wing','left_wing']):
    
    for idx,joint in enumerate(bones):
        marker_dict['color'] = joint.color
        line_dict['color'] = joint.color
        scatter3d(fig,joint.bone.bone_points.cpu(),name[idx],mode = 'lines+markers',line_dict= line_dict)



    
def plot_skeleton_and_skin_normals(skin_points,skeleton,skip_skin_points = 10, normals = False,marker_dict_skeleton = {'size': 10},line_dict_skeleton ={'width': 10}, **kwargs):

    marker_dict_skeleton = {'size': 10}
    line_dict_skeleton = {'width': 10}
    fig = go.Figure()
    plot_cones(fig, skin_points,normals, skip = skip_skin_points,**kwargs)
    plot_skeleton(skeleton,fig,marker_dict_skeleton,line_dict_skeleton)
    return fig

def plot_skin_normals(fig,skin_points,normals,skin,skip_skin_points = 10,color = None, **kwargs):
    plot_cones(fig, skin_points,normals, skip = skip_skin_points,**kwargs)
    return fig

    
def plot_skin(fig,points_to_plot,name,skip_skin_points = 10,color = None,size = 3, **kwargs):
    
    marker_dict_skin = {'size':size,'color': color,  # Set color to distances
            **kwargs}
    scatter3d(fig,points_to_plot[::skip_skin_points,:],name,marker_dict = marker_dict_skin)
    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),aspectmode = 'data',
                             camera_eye=dict(x=1.2, y=1.2, z=0.6)))
    return fig