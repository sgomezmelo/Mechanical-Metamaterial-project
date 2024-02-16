#!/usr/bin/env python
# coding: utf-8

import os
import sys

from dataclasses import dataclass
from enum import Enum

import numpy as np
import nrrd
from skimage import io
from skimage import measure
from scipy.interpolate import interpn
import matplotlib.pyplot as plt

import tomli
io.use_plugin("pil")


#tomlf = os.path.dirname(__file__) + os.sep + "evalconfig.toml"
tomlf = "./evalconfig.toml"
print(tomlf)
with open(tomlf, "rb") as fd:
    tomldict = tomli.load(fd)


# # Parameter config
spacing = tomldict["spacing"] # µm
E = tomldict["young"] # GPa
nu = tomldict["poisson"]

# imask = "/home/bq_jblumberg/Dokumente/MetamaterialsProject/newtry/eval_3/p33_mask.tif"
# dir_a = "/home/bq_jblumberg/Dokumente/MetamaterialsProject/newtry/eval_3/transformix_out"
# outdir = "/home/bq_jblumberg/Dokumente/MetamaterialsProject/newtry/eval_3/sections_p33"

# cmapstr = "seismic"
# cmapstr = "PRGn"

imask = tomldict["imask"]
dir_a = tomldict["defdir"]
outdir = tomldict["outdir"]

cmapstr = tomldict["cmapstr"]


@dataclass
class File:
    name: str
    folder: str

#rawsteps = tomldict["steps"]

steps = [File(name=f"c{i}um", folder=f"{dir_a}/c{i}um") for i in [2,5,9] ]
print(steps)
prefix = "p33"
finite = tomldict["finite"]

def fix_offset(deform):
    umean = np.array([
        np.mean(deform[0,:,:,:]),
        np.mean(deform[1,:,:,:]),
        np.min(deform[2,:,:,-30:-1])
    ])
    
    lmean = np.array([
        np.mean(deform[0,:,:,:4]),
        np.mean(deform[1,:,:,:4]),
        np.max(deform[2,:,:,:4])
    ])
    print("Upper", umean)
    print("Lower", lmean)
    print("Expansion", umean-lmean)
    return deform - lmean[:, None, None, None]

def adjust_deform_to_mask(mask, deform):
    # First crop to large images
    crop = np.maximum(0, np.array(deform.shape[1:]) - np.array(mask.shape))
    cutl = crop // 2
    endr = np.array(deform.shape[1:]) - crop + cutl

    pad = np.maximum(0, np.array(mask.shape) - np.array(deform.shape[1:]))
    padl = pad // 2
    padr = pad - padl

    padw = [(0,0), (padl[0],padr[0]), (padl[1],padr[1]), (padl[2],padr[2])]

    # Print messanges
    for i, n in [(0,"x"), (1,"y"),(2,"z")]:
        if crop[i] > 0:
            # Cropping is allready done, just notify here
            print(f"Warning, cropping image in {n} direction")
        
        if pad[i] > 0:
            print(f"Warning, padding image in {n} direction")

    deformc = deform[:,cutl[0]:endr[0], cutl[1]:endr[1], cutl[2]:endr[2]]
    return np.pad(deformc, pad_width=padw)


def adjust_deformgrad_to_mask(mask, deformgrad):
    # First crop to large images
    crop = np.maximum(0, np.array(deformgrad.shape[2:]) - np.array(mask.shape))
    cutl = crop // 2
    endr = np.array(deformgrad.shape[2:]) - crop + cutl

    pad = np.maximum(0, np.array(mask.shape) - np.array(deformgrad.shape[2:]))
    padl = pad // 2
    padr = pad - padl

    padw = [(0, 0), (0, 0), (padl[0], padr[0]), (padl[1], padr[1]), (padl[2], padr[2])]

    # Print messanges
    for i, n in [(0,"x"), (1,"y"),(2,"z")]:
        if crop[i] > 0:
            # Cropping is allready done, just notify here
            print(f"Warning, cropping image in {n} direction")

        if pad[i] > 0:
            print(f"Warning, padding image in {n} direction")

    deformc = deformgrad[:,:,cutl[0]:endr[0], cutl[1]:endr[1], cutl[2]:endr[2]]
    return np.pad(deformc,pad_width=padw)

def grid_interpolate(verts, tensor):
    #lower = np.floor(verts).astype(np.uint16).T
    #tensor_x = tensor[:,:,1:] - tensor[:,:,:-1]
    #tensor_x = tensor[:,:,1:] - tensor[:,:,:-1]
    points = (np.arange(tensor.shape[2]), np.arange(tensor.shape[3]), np.arange(tensor.shape[4]))
    res = np.empty((3,3,len(verts)))
    for i in range(3):
        for j in range(3):
            res[i,j,:] = interpn(points, tensor[i,j], verts, bounds_error=False)
    return res

def grid_interpolate_vector(verts, tensor):
    #lower = np.floor(verts).astype(np.uint16).T
    #tensor_x = tensor[:,:,1:] - tensor[:,:,:-1]
    #tensor_x = tensor[:,:,1:] - tensor[:,:,:-1]
    points = (np.arange(tensor.shape[1]), np.arange(tensor.shape[2]), np.arange(tensor.shape[3]))
    res = np.empty((3,len(verts)))
    for i in range(3):
        res[i,:] = interpn(points, tensor[i], verts, bounds_error=False)
    return res

def grid_interpolate_scala(verts, field):
    #lower = np.floor(verts).astype(np.uint16).T
    #tensor_x = tensor[:,:,1:] - tensor[:,:,:-1]
    #tensor_x = tensor[:,:,1:] - tensor[:,:,:-1]
    points = (np.arange(field.shape[0]), np.arange(field.shape[1]), np.arange(field.shape[2]))
    return interpn(points, field, verts, bounds_error=False)

def get_defgrad_u(defgrad_f):
    return defgrad_f - np.eye(3)[:,:,None, None, None]

# def get_strain_mag(defgrad_f):
    

def get_strain_from_defgrad(defgrad, finite=False):
    if finite:
        return 0.5 * (
                defgrad + defgrad.swapaxes(0, 1)
                + np.einsum("ki..., kj...->ij...", defgrad, defgrad)
        )
    else:
        return 0.5*(defgrad+defgrad.swapaxes(0,1))

def get_stress_from_strain(E,nu,strain):
    from math import isclose
    twoMu = E/(1.0+nu)
    stress = twoMu*strain
    if not isclose(nu, 0.5):
        trcorr = (stress[0,0]+stress[1,1]+stress[2,2]) * nu / (1-2*nu)
        for i in range(3):
            stress[i,i] += trcorr
    return stress

# def von_miese_stress

def plotstuff(savename, title, mesh, **kwargs):
    p = pv.Plotter(off_screen=True)
    p.set_background("white")
    kwargs["scalar_bar_args"] |= {
        "italic": True,
        # "vertical": True,
        "color": "black"
    }

    single_slice = mesh.slice(normal=[0, 1, 0])
    #p.camera_position = "xz"
    p.add_mesh(single_slice, **kwargs)
    p.screenshot(savename + "_slices.png")

    pv.set_plot_theme("document")
    p.add_mesh(mesh, **kwargs)
    p.screenshot(savename + ".png")
    p.show_axes()
    p.screenshot(savename + "_def.png")
    p.camera_position = "xz"
    p.screenshot(savename + "_xz.png")
    p.camera_position = "xy"
    p.screenshot(savename + "_xy.png")
    p.camera_position = "yz"
    p.screenshot(savename + "_yz.png")
    p.camera_position = "yx"
    cameraxy = p.camera
    cameraxy.roll = 0
    p.reset_camera()
    p.camera = cameraxy
    p.screenshot(savename + "_xyb.png")


def find_parameter_range(sample_exends, p, u, v):
    # Find the parameter range needed to cover the cross section described by
    # vec(x) = vec(p) + s * vec(u) + t * vec(v)

    def fitin(i,j,k, firstend,scndend):
        assert(0<=i<3)
        assert(0<=j<3)

        # matrix = [[u[i], v[i]], [u[j], v[j]]]
        vec_0 = (sample_exends[i] if firstend else 0) - p[i]
        vec_1 = (sample_exends[j] if firstend else 0) - p[j]

        detm = u[i]*v[j]-v[i]*u[j]

        if np.abs(detm) > 0.0001:
            # Unique solution
            # Edge is crossed at a single crossing point
            sr = (v[j]*vec_0-v[i]*vec_1) / detm
            tr = (u[i]*vec_1-u[j]*vec_0) / detm

            # Calculate crossing point along the edge:
            w = p[k] + sr * u[k] + tr * v[k]
            if w < 0 or w > sample_exends[k]:
                # Crossing point lies outside of main area:
                sr, tr = np.nan, np.nan
        else:
            # No unique solution, just return nans
            # This edge is either not reached or the cross section is lying on
            # it. In both cases we
            sr, tr = np.nan, np.nan
        return sr, tr

    # Point  0 lies on the crossing between x=0 and y=0
    # Point  1 lies on the crossing between x=X and y=0
    # Point  2 lies on the crossing between x=X and y=Y
    # Point  3 lies on the crossing between x=0 and y=Y
    # Point  4 lies on the crossing between y=0 and z=0
    # Point  5 lies on the crossing between y=Y and z=0
    # Point  6 lies on the crossing between y=Y and z=Z
    # Point  7 lies on the crossing between y=0 and z=Z
    # Point  8 lies on the crossing between z=0 and x=0
    # Point  9 lies on the crossing between z=Z and x=0
    # Point 10 lies on the crossing between z=Z and x=X
    # Point 11 lies on the crossing between z=0 and x=X

    s = np.zeros(12)
    t = np.zeros(12)

    for ci in range(3):
        cj = (ci+1)%3
        ck = (ci+2)%3
        s[4*ci+0], t[4*ci+0] = fitin(ci, cj, ck, False, False)
        s[4*ci+1], t[4*ci+1] = fitin(ci, cj, ck, True, False)
        s[4*ci+2], t[4*ci+2] = fitin(ci, cj, ck, True, True)
        s[4*ci+3], t[4*ci+3] = fitin(ci, cj, ck, False, True)

    # Find the number of crossing points with the image area
    npoints = np.count_nonzero(~np.isnan(s))

    smin = np.nanmin(s)
    smax = np.nanmax(s)
    tmin = np.nanmin(t)
    tmax = np.nanmax(t)
    
    # There are now a few scenarios:
    if npoints < 3:
        # The requested cross section plane does not cross the interior of the
        # imaged area
        return npoints, np.nan, np.nan, np.nan, np.nan
    elif npoints == 3:
        # The requested cross section plane describes a triangle.
        return 3, smin, smax, tmin, tmax
    elif npoints == 4:
        # The requested cross section plane describes a parallelogram
        return 4, smin, smax, tmin, tmax
    else:
        # This is impossible
        return 4, smin, smax, tmin, tmax
        #raise RuntimeError("Internal error")


    # a) The are 0 crossing points -> No


class SectionProjector:
    def __init__(self, mask, p, u, v):
        # Plots the crosssection described by the plain, parameterized by
        # vec(x) = vec(p) + s * vec(u) + t * vec(v)

        p = np.asanyarray(p)
        u = np.asanyarray(u)
        v = np.asanyarray(v)

        self.p = p
        self.u = u
        self.v = v

        self.shape = mask.shape

        sample_range = np.array(mask.shape) * spacing

        ncorners, smin, smax, tmin, tmax = find_parameter_range(sample_range, p, u, v)
        if smin == np.nan:
            raise RuntimeError("No crossing plane")

        self.lenu = np.sqrt(np.sum(u**2))
        self.lenv = np.sqrt(np.sum(v**2))

        self.n = np.cross(u, v) / (self.lenu*self.lenv)

        srng = np.arange(smin-spacing, smax+spacing, spacing/self.lenu)
        trng = np.arange(tmin-spacing, tmax+spacing, spacing/self.lenv)

        S, T = np.meshgrid(srng, trng)

        XYZ = np.empty((S.shape[0],S.shape[1],3))

        for i in range(3):
            XYZ[:,:,i] = p[i] + S * u[i] + T * v[i]

        self.s = S
        self.t = T

        xrng = np.arange(mask.shape[0]) * spacing
        yrng = np.arange(mask.shape[1]) * spacing
        zrng = np.arange(mask.shape[2]) * spacing
        self.rngs = (xrng, yrng, zrng)
        self.maski = (interpn(self.rngs, 1.*mask, XYZ, method="nearest", bounds_error=False, fill_value=0.) != 0.)
        self.xyz = XYZ
        print(f"Create new projector with {ncorners} courners and s in [{smin},{smax}], t in [{tmin},{tmax}]")

    def get_projection_image(self, field):
        assert(field.shape == self.shape)
        fieldi = interpn(self.rngs, field, self.xyz, method="linear", bounds_error=False, fill_value=0.)

        fieldi[~self.maski] = np.nan
        return fieldi

    def get_vector_projection(self, fieldx, fieldy, fieldz):
        fieldxi = self.get_projection_image(fieldx)
        fieldyi = self.get_projection_image(fieldy)
        fieldzi = self.get_projection_image(fieldz)
        fieldu = (fieldxi * self.u[0] + fieldyi * self.u[1] + fieldzi * self.u[2]) / self.lenu
        fieldv = (fieldxi * self.v[0] + fieldyi * self.v[1] + fieldzi * self.v[2]) / self.lenv
        fieldn = (fieldxi * self.n[0] + fieldyi * self.n[1] + fieldzi * self.n[2])
        return fieldu, fieldv, fieldn

    @staticmethod
    def _h_decode(vec, norm, lbl):
        if (vec == np.array([1., 0., 0.])).all():
            return "x"
        elif (vec == np.array([0., 1., 0.])).all():
            return "y"
        elif (vec == np.array([0., 0., 1.])).all():
            return "z"
        elif (vec == np.array([0., 0., 0.])).all():
            raise RuntimeError
        else:
            string = lbl + " "
            step = "="
            if vec[0] != 0.:
                string += f" {step} {vec[0] / norm:.2f}$e_x$ "
                step = "+"
            if vec[1] != 0.:
                string += f" {step} {vec[1] / norm:.2f}$e_y$ "
                step = "+"
            if vec[2] != 0.:
                string += f" {step} {vec[2] / norm:.2f}$e_z$ "
                step = "+"
            assert(step == "+")
        return string

    def get_lbl_u(self):
        return self._h_decode(self.u, self.lenu, "u")

    def get_lbl_v(self):
        return self._h_decode(self.v, self.lenv, "v")

def bg_quiver(axes, X, Y, U, V, enspare=1, quiverargs={}, imshowargs={}):
    offsx = (U.shape[0] % enspare) // 2
    offsy = (U.shape[1] % enspare) // 2

    rXs = np.arange(offsx, U.shape[0], enspare)
    rYs = np.arange(offsy, U.shape[1], enspare)

    Xs, Ys = np.meshgrid(rXs, rYs, indexing="ij")
    Us = U[offsx::enspare, offsy::enspare]
    Vs = V[offsx::enspare, offsy::enspare]

    amp = np.sqrt(U*U+V*V)
    im = axes.imshow(amp, origin="lower", **imshowargs)
    axes.quiver(Ys.T, Xs.T, Us.T, Vs.T, color="w", **quiverargs)
    return im

def plot_crossection(axes, projection, field, **kwargs):
    #axes.invert_yaxis()
    return axes.imshow(projection.get_projection_image(field), origin="upper", **kwargs)

def plot_crosssection_vector_tg(axes, projection, fieldx, fieldy, fieldz, enspare=10, quiverargs={}, **imshowargs):
    fieldu, fieldv, fieldn = projection.get_vector_projection(fieldx, fieldy, fieldz)
    return bg_quiver(axes, projection.s, projection.t, fieldu, fieldv, enspare, quiverargs, imshowargs)


def setup_projections(mask):
    #centers = mask.cell_centers()
    #print(mask)
    # Detect a point well within the sample. You will probably have
    # to change it for each sample
    # This is currently optimized for the cube like ones
    xslope = np.count_nonzero(1*mask, axis=(1,2))
    yslope = np.count_nonzero(1*mask, axis=(0,2))
    zslope = np.count_nonzero(1*mask, axis=(0,1))

    xpos = spacing * (np.argmax(xslope > 0.7 * np.max(xslope)) + 2)
    ypos = spacing * (np.argmax(yslope > 0.7 * np.max(yslope)) + 2)
    zpos = spacing * (np.argmax(zslope > 0.7 * np.max(zslope)) + 2)

    n, m, k = mask.shape
    polar_angle = (-5.71)*np.pi/180.0
    d_spacing = -16.0
    p0 = np.array([n, m, k-6])*spacing/2.0 + np.array([np.sin(polar_angle),-np.cos(polar_angle),0.0])*d_spacing*spacing
    #print(p0)
    #p0 = [spacing, 0.0, 0.0]
    plist = {
        "xz": SectionProjector(
            mask=mask,
            p=p0, # Base point
            u=[1.,0.,0.], # Spanning vector 1
            v=[0.,0.,1.], # Spanning vector 2
        ),
        "xy": SectionProjector(
            mask=mask,
            p=p0,
            u=[1., 0., 0.],
            v=[0., 1., 0.],
        ),
        "yx": SectionProjector(
            mask=mask,
            p=p0,
            u=[1., 0., 0.],
            v=[0., 1., 0.],
        ),
        "yz": SectionProjector(
            mask=mask,
            p=p0,
            u=[0., 1., 0.],
            v=[0., 0., 1.],
        ),
        "xz2": SectionProjector(
            mask=mask,
            p=[p0[0], p0[1], spacing * mask.shape[2] / 2],
            u=[1., 0., 0.],
            v=[0., 0., 1.],
        ),
        "diag": SectionProjector(
            mask=mask,
            p=p0,
            u=[np.cos(polar_angle), np.sin(polar_angle) , 0.],
            v=[0., 0., 1.],
        ),
    }

    #Add projection to each plane to calculate average z displacement
    dn_plane = 5

    for i in range(10):
        pnum = dn_plane*i+120
        plist["xy_plane_"+str(pnum)]  = SectionProjector(
            mask=mask,
            p=[0., 0., pnum*spacing],
            u=[1., 0., 0.],
            v=[0., 1., 0.],
        )
    return plist

def plot_projections_scalar(savename, title, plist, dataset, Xmin, Xmax):
    cmap = plt.cm.get_cmap(cmapstr).copy()
    cmap.set_bad('black', 1.)

    for key, proj in plist.items():
        fig, ax = plt.subplots()
        pos = plot_crossection(ax, proj, dataset, cmap=cmap, vmin = Xmin, vmax = Xmax)
        fig.colorbar(pos)
        ax.set_aspect("equal")
        ax.set_xlabel(proj.get_lbl_u())
        ax.set_ylabel(proj.get_lbl_v())
        ax.set_title(title)
        fig.savefig(savename + f"_{key}.png")
        x = proj.get_projection_image(dataset)
        n, m= x.shape
        i = int(n/2)
        j = int(m/2)
        n = 65
        m = 130
        #mean_field = np.mean(x[~np.isnan(x)])
        #print("The mean of "+title+" in projection "+key+" is "+str(mean_field))
        if key == "diag":
            # print("single point at (77,430)", x[77,430])
            # print("single point at (77,440)", x[77,440])
            # print("single point at (77,450)", x[77,450])
            # print("single point at (77,480)", x[77,480])
            for di  in [-1,0,1]:
                print("Looking at plane " +str(di*2+2))
                for dj in range(3):
                    print("The "+title+"of file "+ key +" at point ("+str(i+m*di)+", "+str(j+n*dj)+") is "+str(x[i+m*di,j+n*dj]))
            
            for di  in [0,1]:
                print("Looking at plane " +str(di*2+1))
                for dj in range(3):
                    print("The "+title+"of file "+ key +" at point ("+str(i-int(m/2)+m*di)+", "+str(j+10+int(n/2)+n*dj)+") is "+str(x[i-int(m/2)+m*di,j+10+int(n/2)+n*dj]))
        plt.close(fig)

def plot_projections_vector(savename, title, plist, dataset):
    cmap = plt.cm.get_cmap(cmapstr).copy()
    cmap.set_bad('black', 1.)

    for key, proj in plist.items():
        fig, ax = plt.subplots()
        pos = plot_crosssection_vector_tg(ax, proj, dataset[0], dataset[1], dataset[2], cmap=cmap)
        fig.colorbar(pos)
        ax.set_aspect("equal")
        ax.set_xlabel(proj.get_lbl_u())
        ax.set_ylabel(proj.get_lbl_v())
        ax.set_title(title)
        fig.savefig(savename + f"_{key}.png")
        plt.close(fig)

def plot_views(savename, title, mesh, **kwargs):
    p = pv.Plotter(off_screen=True)
    p.set_background("white")
    kwargs["scalar_bar_args"] |= {
        "italic": True,
        # "vertical": True,
        "color": "black"
    }
    pv.set_plot_theme("document")
    p.add_mesh(mesh, **kwargs)
    #p.screenshot(savename + ".png")
    #p.show_axes()
    #p.screenshot(savename + "_def.png")
    p.camera_position = "xz"
    p.screenshot(savename + "_xz.png")
    p.camera_position = "xy"
    p.screenshot(savename + "_xy.png")
    p.camera_position = "yz"
    p.screenshot(savename + "_yz.png")
    p.camera_position = "yx"
    cameraxy = p.camera
    cameraxy.roll = 0
    p.reset_camera()
    p.camera = cameraxy
    p.screenshot(savename + "_xyb.png")

    # p.save_graphic(savename + ".svg", title=title)

def load_mask(maskpath, trim_surfaces = True):
    img = io.imread(maskpath).transpose(2, 1, 0)

    print("Default orientation")
    img = img[:, :, ::-1] # Images get inverted in elastix

    if trim_surfaces:
        img[0,:,:] = 0
        img[-1,:,:] = 0
        img[:,0,:] = 0
        img[:,-1,:] = 0

    verts, faces, normals, values = measure.marching_cubes(img)

    faces_b = np.hstack((3 * np.ones((faces.shape[0], 1), dtype=int), faces)).ravel()

    return img, verts, faces_b, normals


def load_deform(pathbase):
    path2 = pathbase + os.sep + "deformationField.nrrd"

    deform, dheader = nrrd.read(path2)
    deform *= spacing

    # if inverted: # Invert z orientation
    #    pass
    #    # deform = deform[:,:,::-1]
    #    # deform[2,:,:] *= -1

    deform = fix_offset(deform)

    return deform

def load_defgrad(pathbase):
    path3 = pathbase + os.sep + "fullSpatialJacobian.nrrd"

    jacobi, jheader = nrrd.read(path3)
    defgrad = np.reshape(jacobi, (3, 3) + jacobi.shape[1:])

    return defgrad

def ensure_dir(dirname):
    if not os.path.exists(dirname):
        print("Create dir", dirname)
        os.mkdir(dirname)

def get_straintr_s(strain_s):
    strainmag = strain_s[0, 0] + strain_s[1, 1] + strain_s[2, 2]
    return strainmag

def analyse_step(filea: File):
    umax = tomldict["umax"]
    umin = tomldict["umin"]
    umin_x = tomldict["umin_x"]
    umin_y = tomldict["umin_y"]
    umax_x = tomldict["umax_x"]
    umax_y = tomldict["umax_y"]
    emin = tomldict["emin"]
    emax = tomldict["emax"]
    tmax = tomldict["tmax"]

    img, verts, faces, normals = load_mask(imask, trim_surfaces=False)
    print("Step 1 done")
    plist = setup_projections(img)
    print("Setup Down")
    # mesh = pv.PolyData(verts, faces)

    print(f"Load dataset {filea.folder} ...")
    deform = adjust_deform_to_mask(img, load_deform(filea.folder))
    defgrad_f = adjust_deformgrad_to_mask(img, load_defgrad(filea.folder))
    defgrad_u = get_defgrad_u(defgrad_f)
    defgrad_f = None
    #outpath = os.path.dirname(os.path.dirname(pathbase)) + os.sep + "plots" + os.sep + stepfile
    outpath = outdir

    ensure_dir(os.path.dirname(outpath))
    ensure_dir(outpath)
    print("Load completed")

    print("Compute strain fields")
    strain = get_strain_from_defgrad(defgrad_u, finite=finite)
    strainmag = get_straintr_s(strain)
    
    print("Compute stress fields")
    stress = get_stress_from_strain(E, nu, strain)
    
    

    print("Calculation completed")
    #mesh = pv.PolyData(verts, faces)
    #mesh.point_data["strainmag"] = strainmag
    #mesh.point_data["traction"] = traction_ampl
    #mesh.point_data["traction_z"] = traction[:, 2]
    #mesh.point_data["strainzz"] = strain_s[2, 2]
    #mesh.point_data["displ_x"] = grid_interpolate_scala(verts, deform[0])
    #mesh.point_data["displ_y"] = grid_interpolate_scala(verts, deform[1])
    #mesh.point_data["displ_z"] = grid_interpolate_scala(verts, deform[2])
    #mesh.point_data["displzz"] = grid_interpolate_scala(verts, defgrad[2, 2])
    # pv.set_plot_theme('default')

    fnamebase = outpath + os.sep + filea.name

    # rstring = f"difference between {str(str(filea.type_))} {filea.name} and {str(str(fileb.type_))} {fileb.name}"

    plot_projections_vector(
        fnamebase + "displt",
        f'Tangential Displacement for {filea.name} (um)',
        plist,
        deform
    )

    plot_projections_scalar(
        fnamebase + "displz",
        f'z Displacement for {filea.name} (um)',
        plist,
        deform[2], 
        umin, 
        umax

    )

    plot_projections_scalar(
        fnamebase + "displx",
        f'x Displacement for {filea.name} (um)',
        plist,
        deform[0], 
        umin_x, 
        umax_x

    )

    plot_projections_scalar(
        fnamebase + "disply",
        f'y Displacement for {filea.name} (um)',
        plist,
        deform[1], 
        umin_y, 
        umax_y

    )

    # plot_projections_scalar(
    #     fnamebase + "strainmag",
    #     f'z Displacement for {filea.indent} um, scan {filea.scannr}',
    #     plist,
    #     strainmag,
    # )

    plot_projections_scalar(
        fnamebase + "strainzz",
        f'Ezz for {filea.name} in %',
        plist,
        100*strain[2,2],
        emin,
        emax
    )
    
    plot_projections_scalar(
       fnamebase + "stresszz",
        f'Tzz for {filea.name} in GPa',
        plist,
        stress[2,2],
        -tmax,
        tmax
    )

import sys

if len(sys.argv) > 1:
    snr = int(sys.argv[1]) - 1
    s1 = steps[snr]
    analyse_step(s1)
else:
    for s1 in steps[0:]:
        analyse_step(s1)



