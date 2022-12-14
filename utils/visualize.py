from PIL import Image, ImageDraw
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import cv2
def visualize_attn(attn_map):
    return None

def grid_show(to_shows, cols):
    rows = (len(to_shows)-1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows*8.5, cols*2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()

def visualize_head(att_map,idx,i):
    att_map=att_map.permute(0,1,4,2,3)
    att_map = torch.unbind(att_map, dim=0)
    for id,xb in enumerate(att_map):
        xs = torch.unbind(xb, dim=0)
        for frame,x in enumerate(xs):
            x=x.cpu().numpy()[0]
            ax = plt.gca()
            # Plot the heatmap
            im = ax.imshow(x)
            # Create colorbar
            #cbar = ax.figure.colorbar(im, ax=ax)
            #plt.show()
            plt.savefig('output/{}_{}_batchid_{}_frame{}.jpg'.format(idx,i,id,frame))


def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)


def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a = ImageDraw.ImageDraw(image)
        a.rectangle([(y * w, x * h), (y * w + w, x * h + h)], fill=None, outline='red', width=2)
    return image


def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=0.6):
    att_maps = torch.unbind(att_map.permute(0, 1, 4, 2, 3),dim=0)
    images = torch.unbind(image, dim=0)
    for i,att_mapv in enumerate(att_maps):
        imagev=torch.unbind(images[i],dim=0)
        att_mapv = torch.unbind(att_mapv, dim=0)
        for j,frame in enumerate(att_mapv):
            frame=frame.cpu().numpy()[0]
            image=imagev[j].cpu().numpy()[0]
            if not isinstance(grid_size, tuple):
                grid_size = (grid_size, grid_size)

            H, W = frame.shape
            with_cls_token = False

            grid_image = highlight_grid(image, [grid_index], grid_size)

            mask = frame[grid_index].reshape(grid_size[0], grid_size[1])
            mask = Image.fromarray(mask).resize((image.size))

            fig, ax = plt.subplots(1, 2, figsize=(10, 7))
            fig.tight_layout()

            ax[0].imshow(grid_image)
            ax[0].axis('off')

            ax[1].imshow(grid_image)
            ax[1].imshow(mask / np.max(mask), alpha=alpha, cmap='rainbow')
            ax[1].axis('off')
            plt.show()


def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H / grid_size[0])
    delta_W = int(W / grid_size[1])

    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]

    padded_image = np.hstack((padding, image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W / 4), int(delta_H / 4)), 'CLS', fill=(0, 0, 0))  # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask), cls_weight)
    cls_weight = cls_weight / max(np.max(mask), cls_weight)

    if len(padding.shape) == 3:
        padding = padding[:, :, 0]
        padding[:, :] = np.min(mask)
    mask_to_pad = np.ones((1, 1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H, :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask

    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1], 4))
    meta_mask[delta_H:, 0: delta_W, :] = 1

    return padded_image, padded_mask, meta_mask


def visualize_grid_to_grid_with_cls(att_map, grid_index, image, batch_idx,view,label,grid_size=14, alpha=0.3):
    att_maps = torch.unbind(att_map.permute(0, 1, 4, 2, 3),dim=0) #B T h H W
    images = torch.unbind(image, dim=0)
    # for i,att_mapv in enumerate(att_maps):
    #     imagev=torch.unbind(images[i],dim=0)
    #     att_mapv = torch.unbind(att_mapv, dim=0)
    #
    #     fig, ax = plt.subplots(4, 4, figsize=(9,9))
    #     fig.tight_layout()
    #     for j,frame in enumerate(att_mapv):
    #         frame=frame.cpu().numpy().mean(0) #head avg
    #         image=imagev[j].cpu()
    #         if not isinstance(grid_size, tuple):
    #             grid_size = (grid_size, grid_size)
    #
    #         attention_map = frame[grid_index]
    #         cls_weight = attention_map
    #
    #         mask = cls_weight.reshape(grid_size[0], grid_size[1])
    #         mask = cv2.resize(mask,(224,224))
    #         mask = mask / np.max(mask)
    #         mask = (mask * 255).astype('uint8')
    #         image =  image.permute(1,2,0) * torch.tensor([58.395, 57.12, 57.375]) /255.0 + torch.tensor([123.675, 116.28, 103.53])/255.0
    #         image = image.permute(2,1,0)
    #         image=transforms.ToPILImage()(image).convert('RGB').rotate(270).transpose( Image.FLIP_LEFT_RIGHT)
    #         ax[int(j/4),int(j%4)].imshow(image,alpha=1)
    #         ax[int(j / 4), int(j % 4)].axis('off')
    #         ax[int(j/4),int(j%4)].imshow(mask, alpha=alpha, interpolation='nearest',cmap='jet')
    #         ax[int(j/4),int(j%4)].axis('off')
    #
    #     plt.savefig('output2/grid{}__idx{}_in_batch{}_view_{}.jpg'.format(grid_index,i,batch_idx,view))
    #     plt.close()
    for i,att_mapv in enumerate(att_maps):
        imagev=torch.unbind(images[i],dim=0)
        att_mapv = torch.unbind(att_mapv, dim=0)

        fig, ax = plt.subplots(4, 4, figsize=(9,9))
        fig.tight_layout()
        for j,frame in enumerate(att_mapv):
            frame=frame.cpu().numpy().mean(0) #head avg
            image=imagev[j].cpu()
            if not isinstance(grid_size, tuple):
                grid_size = (grid_size, grid_size)

            attention_map = frame[grid_index+5,9:]
            cls_weight = attention_map

            mask = cls_weight.reshape(grid_size[0], grid_size[1])
            mask = cv2.resize(mask,(224,224))
            mask = mask / np.max(mask)
            mask = (mask * 255).astype('uint8')
            image =  image.permute(1,2,0) * torch.tensor([58.395, 57.12, 57.375]) /255.0 + torch.tensor([123.675, 116.28, 103.53])/255.0
            image = image.permute(2,1,0)
            image=transforms.ToPILImage()(image).convert('RGB').rotate(270).transpose( Image.FLIP_LEFT_RIGHT)
            ax[int(j/4),int(j%4)].imshow(image,alpha=1)
            ax[int(j / 4), int(j % 4)].axis('off')
            ax[int(j/4),int(j%4)].imshow(mask, alpha=alpha, interpolation='nearest',cmap='jet')
            ax[int(j/4),int(j%4)].axis('off')

        plt.savefig('2dscls_out2/{}_{}_{}_label{}_grid{}.jpg'.format(i,batch_idx,view,label[i],grid_index+98))
        plt.close()

def visualize_grid_to_grid_with_cls_th(att_map, grid_index, image, batch_idx,view,grid_size=14, alpha=0.3): # 2 14 224 224 12
    att_maps = torch.unbind(att_map.permute(0, 1, 4, 2, 3),dim=0) #B T h H W
    images = torch.unbind(image, dim=0)
    for i,att_mapv in enumerate(att_maps):# 14 12 224 224
        att_map = att_mapv.mean(1) # 14 224 224 # head avg
        attention_maps = att_map[:,grid_index].reshape(14,16,14).softmax(dim=-1).permute(1,0,2).cpu().numpy() # 14 224
        attention_maps= attention_maps.reshape(16,196)
        #attention_maps = torch.unbind(attention_maps, dim=0)

        imagev = torch.unbind(images[i], dim=0)
        fig, ax = plt.subplots(4, 4, figsize=(9,9))
        fig.tight_layout()
        for j in range(16):
            image=imagev[j].cpu()
            if not isinstance(grid_size, tuple):
                grid_size = (grid_size, grid_size)

            cls_weight = attention_maps[j]

            mask = cls_weight.reshape(grid_size[0], grid_size[1])
            mask = cv2.resize(mask,(224,224))
            mask = mask / np.max(mask)
            mask = (mask * 255).astype('uint8')
            image =  image.permute(1,2,0) * torch.tensor([58.395, 57.12, 57.375]) /255.0 + torch.tensor([123.675, 116.28, 103.53])/255.0
            image = image.permute(2,1,0)
            image=transforms.ToPILImage()(image).convert('RGB').rotate(270).transpose( Image.FLIP_LEFT_RIGHT)
            ax[int(j/4),int(j%4)].imshow(image,alpha=1)
            ax[int(j / 4), int(j % 4)].axis('off')
            ax[int(j/4),int(j%4)].imshow(mask, alpha=alpha, interpolation='nearest',cmap='jet')
            ax[int(j/4),int(j%4)].axis('off')

        plt.savefig('output/TH_grid{}__idx{}_in_batch{}_view_{}.jpg'.format(grid_index,i,batch_idx,view))
        plt.close()

def visualize_data(image, idx,m,grid_size=14):
    #image = image.permute(0, 2, 1, 3, 4)
    images = torch.unbind(image, dim=0)
    for i,imagev in enumerate(images):
        imagev=torch.unbind(imagev, dim=0)
        fig, ax = plt.subplots(4, 4, figsize=(10,10))
        fig.tight_layout()
        for j,frame in enumerate(imagev):
            frame=frame.cpu()
            if not isinstance(grid_size, tuple):
                grid_size = (grid_size, grid_size)

            frame =  frame.permute(1,2,0) * torch.tensor([58.395, 57.12, 57.375]) /255.0 + torch.tensor([123.675, 116.28, 103.53])/255.0
            frame = frame.permute(2,1,0)
            frame=transforms.ToPILImage()(frame).convert('RGB').rotate(270)

            ax[int(j/4),int(j%4)].imshow(frame,alpha=1)
            ax[int(j / 4), int(j % 4)].axis('off')

        plt.savefig('output/{}_idx{}_i_{}.jpg'.format(i,idx,m))
        plt.close()