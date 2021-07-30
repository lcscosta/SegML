#   Processamento de Imagens Médicas FFCLRP - USP
#   Prof. Murta
#
#   Lucas Murilo da Costa - lucasdacosta@usp.br
#   Luiza Licarião - 
#
###########################################################
#          Script p/ rodar GMM                            #
###########################################################

# Load images
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import scipy.stats as stats
from sys import argv
import os

def loadimages(name):
    im = Image.open(name)
    dat = np.asarray(im)
    return im, dat

def create_hist(dat):
	num, bins, patches = plt.hist(dat.flatten(), 255, density=True, facecolor='g', alpha=0.75)
	return num

def run_gmm(hist):
	lowest_bic = np.infty
	bic = []
	bestcompo = []
	bestcv = []

	n_components_range = range(1, 7)
	cv_types = ['spherical', 'tied', 'diag', 'full']

	s = hist.reshape(-1,1)
	for n_component in n_components_range:
		for cv_type in cv_types:
			# Fit a mixture of Gaussians with EM
			gmm = mixture.GaussianMixture(n_components=n_component, covariance_type=cv_type)
			gmm.fit(s)
			bic.append(gmm.bic(s))
			if bic[-1] < lowest_bic:
				lowest_bic = bic[-1]
				best_cv=cv_type
				best_ncompo = n_component
				best_gmm = gmm

	weights = best_gmm.weights_
	means = best_gmm.means_
	covars = best_gmm.covariances_
	
	data_gmm = []
	if best_cv == "spherical":
		for i in range(best_ncompo):
			data_gmm.append([hist*weights[i]*stats.norm.pdf(hist,means[i],np.sqrt(covars[i]))])
	else:
		for i in range(best_ncompo):
			data_gmm.append([hist*weights[i]*stats.norm.pdf(hist,means[i],np.sqrt(covars[i][0]))])
        
	data_gmm = np.array(data_gmm).reshape(best_ncompo,255)
	
	return data_gmm

def gray_gmm(data_gmm):
	indices = []
	for i in range(len(data_gmm)):
		ind, = np.where(data_gmm[i] == i)
		indices.append(ind)
		
	return indices

def seg(im, indices):
	width, height = im.size
	for i in range(len(indices)):		
		new_image = Image.new(mode = "L", size = (width,height), color = 255)
		dat = np.asarray(im)
		for j in range(height):
			for k in range(width):
				for l in range(len(indices[i])):
					if len(indices[i]) > 0 and dat[j][k][0] == indices[i][l]:
						new_image.putpixel((k, j),0)
									   
		new_image.save("output_"+str(i)+".jpeg", 'jpeg')

			
def main(argv):
	#passos:
	print('flag')
	os.chdir(os.getcwd()+'/images')
	#1. abrir img
	im, dat = loadimages(argv[0])
	#2. criar histograma
	hist = create_hist(dat)
	#3. fazer gmm
	data_gmm = run_gmm(hist)
	#4. criar regioes de dominio do gmm
	grays = gray_gmm(data_gmm)
	#5. para cada região exportar uma segmentação
	seg_img = seg(im, grays)
	
	print('finalizado')
	
main(argv[1:])
