#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:49:08 2022

@author: sg
"""
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt

def read_gnuplot_file(filename):
    assert isinstance(filename, str)
    assert filename.endswith('.gpl'), '{0} does not end with .gpl'.format(filename)
    assert path.exists(filename), '{0} does not exist'.format(filename)
    data = np.loadtxt(filename, dtype=float)
    mask = np.sum(np.diff(data, axis=0), axis=1) != 0.0
    mask = np.concatenate((np.array([True]), mask))
    data = data[mask]
    positions = data[:, 0]
    values = data[:, 1]
    return positions, values

def extract_time(filename):
    assert isinstance(filename, str)
    assert filename.endswith('.gpl'), '{0} does not end with .gpl'.format(filename)
    assert path.exists(filename), '{0} does not exist'.format(filename)
    directory, filename = path.split(filename)
    split = filename.strip('.gpl').split('-')
    assert len(split) > 2
    filename = str('-').join(split[:2])
    filename += '.txt'
    filename = path.join(directory, filename)
    assert path.exists(filename), '{0} does not exist'.format(filename)
    step = int(split[-1])
    assert step >= 0
    data = np.loadtxt(filename, usecols=(1, 2), delimiter='|', skiprows=1)
    steps = data[:,0].astype(int)
    time = data[steps == step, 1]
    return float(time)

def compute_exact_solution(x, time):
    assert isinstance(x, np.ndarray)
    assert isinstance(time, float)
    
    velocity = 1.0
    ell = 1.0
    n = np.floor(velocity * time / ell)
    
    xi = np.copy(x)
    if n > 0.0:
        xi -= velocity * time / (n * ell)
    elif time != 0.0:
        xi -= velocity * time
    assert xi.min() >= -ell
    assert xi.max() <= ell
    
    xi[xi < 0.0] += ell

    assert xi.min() >= 0.0
    assert xi.max() <= ell
    
    solution = np.zeros_like(xi, dtype=float)

    mask = np.abs(2.0 * xi - 0.3) <= 0.25
    eta = 2.0 * xi[mask] - 0.3
    solution[mask] = np.exp(-300.0 * eta**2)
    
    mask = np.abs(2.0 * xi - 0.9) <= 0.2
    solution[mask] = 1.0
    
    mask = np.abs(2.0 * xi - 1.6) <= 0.2
    eta = 2.0 * xi[mask] - 1.6
    solution[mask] = np.sqrt(1.0 - (eta / 0.2)**2)

    return solution
    
def create_plot(filename, save_png=True):
    assert isinstance(filename, str)
    filename.endswith('.gpl')
    x, numerical_solution = read_gnuplot_file(filename)
    time = extract_time(filename)
    exact_solution = compute_exact_solution(x, time)
    fig = plt.figure()
    plt.plot(x, exact_solution, 'r-', x, numerical_solution, 'b--x')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(which='both')
    filename = filename.replace('.gpl', '.png')
    plt.savefig(filename, dpi=600)
    plt.close(fig)
    
def save_movie(directory, fps=20):
    files = [path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')]
    directory, filename = path.split(files[0])
    split = filename.strip('.png').split('-')
    assert len(split) > 2
    filename = str('-').join(split[:2])
    filename +='-*'
    filename += '.png'
    filename += "'"
    filename = "'" + filename
    file_pattern = path.join(directory, filename)
    print(file_pattern)
    _, filename = path.split(directory)
    filename += '.mp4'
    os.system('ffmpeg -r {0} -pattern_type glob  -i {1} -vcodec mpeg4 -y {2}'.format(fps, file_pattern, filename))
    
if __name__ == "__main__":
    methods = ('bogacki_shampine',
               'cash_karp',
               'dopri',
               'fehlberg',
               'heun_euler',
               'rk3',
               'rk4')
    cwd = os.getcwd()
    for method in methods:
        directory = path.join(cwd, method)
        files = [path.join(directory, f) for f in os.listdir(directory) if f.endswith('.gpl')]
        for filename in files:
            print(filename)
            create_plot(filename)
        save_movie(directory)