# Variational Auto-Encoder
Paper Reimplementation —— "D. P. Kingma and M. Welling. [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114). *ICLR*, 2014."

### Dependencies
```
python         version-3.6.7
pytorch        version-0.4.1
matplotlib     version-3.1.1
numpy          version-1.17.4
scipy          version-1.3.2
```

### Architecture
```
For MNIST:
          x(784) -> h1(200) -> z( 
          
For FreyFace:
          x(560) -> h1
```

### Experiments
According to the



#### Likelihood lower bound


#### Generation
<table align='center'>
<tr align='center'>
<td> Generated Process </td>
<td> 2-D latent space </td>
<td> 5-D latent space </td>
<td> 10-D latent space </td>
<td> 20-D latent space </td>
</tr>
<tr>
<td><img src = 'results/MNIST/samples-gif.gif' height = '150px'>
<td><img src = 'results/MNIST/gene_imgs-2D.png' height = '150px'>
<td><img src = 'results/MNIST/gene_imgs-5D.png' height = '150px'>
<td><img src = 'results/MNIST/gene_imgs-10D.png' height = '150px'>
<td><img src = 'results/MNIST/gene_imgs-20D.png' height = '150px'>
</tr>
<tr>
<td><img src = 'results/FreyFace/samples-gif.gif' height = '150px'>
<td><img src = 'results/FreyFace/gene_imgs-2D.png' height = '150px'>
<td><img src = 'results/FreyFace/gene_imgs-5D.png' height = '150px'>
<td><img src = 'results/FreyFace/gene_imgs-10D.png' height = '150px'>
<td><img src = 'results/FreyFace/gene_imgs-20D.png' height = '150px'>
</tr>
</table>

### Reconstruction

<table align='center'>
<tr align='center'>
<td> Ground Truth </td>
<td> 2-D latent space </td>
<td> 5-D latent space </td>
<td> 10-D latent space </td>
<td> 20-D latent space </td>
</tr>
<tr>
<td><img src = 'results/MNIST/origin_imgs.png' height = '150px'>
<td><img src = 'results/MNIST/reconst_imgs-2D.png' height = '150px'>
<td><img src = 'results/MNIST/reconst_imgs-5D.png' height = '150px'>
<td><img src = 'results/MNIST/reconst_imgs-10D.png' height = '150px'>
<td><img src = 'results/MNIST/reconst_imgs-20D.png' height = '150px'>
</tr>
</table>

<table align='center'>
<tr align='center'>
<td> Ground Truth </td>
<td> 2-D latent space </td>
<td> 5-D latent space </td>
<td> 10-D latent space </td>
<td> 20-D latent space </td>
</tr>
<tr>
<td><img src = 'results/FreyFace/origin_imgs.png' height = '150px'>
<td><img src = 'results/FreyFace/reconst_imgs-2D.png' height = '150px'>
<td><img src = 'results/FreyFace/reconst_imgs-5D.png' height = '150px'>
<td><img src = 'results/FreyFace/reconst_imgs-10D.png' height = '150px'>
<td><img src = 'results/FreyFace/reconst_imgs-20D.png' height = '150px'>
</tr>
</table>

### Manifold

<table align='center'>
<tr align='center'>
<td> Learned MNIST manifold </td>
<td> Distribution of labeled data  </td>
</tr>
<tr>
<td><img src = 'results/MNIST/MNIST-manifold1.png' height = '400px'>
<td><img src = 'results/MNIST/latent_distribution.png' height = '400px'>
</tr>
</table>



## References
The implementation is based on the projects:  
[1] https://github.com/pytorch/examples  
[2] https://github.com/yunjey/pytorch-tutorial

## Acknowledgements
This implementation has been tested with PyTorch r0.4.1 on Windows 10.
