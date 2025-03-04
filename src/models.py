import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .networks import SEM, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, TVLoss
import math


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)
            
            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else: 
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)



class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)


        generator = SEM()
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

        #### learning rate decay
        self.gen_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.gen_optimizer, last_epoch=-1, milestones=[20000, 40000,60000,80000,120000], gamma=self.config.LR_Decay)
        self.dis_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.dis_optimizer, last_epoch=-1,
                                                                  milestones=[20000, 40000,60000,80000,120000], gamma=self.config.LR_Decay)
                                                                  
                                                                  
        self.scaler = torch.cuda.amp.GradScaler()

        

    def process(self, images, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs

        outputs_img = self(images, masks)

        
        gen_loss = 0
        dis_loss = 0
        
        
        with torch.cuda.amp.autocast():


        # discriminator loss
            dis_input_real = images
            dis_input_fake = outputs_img.detach()


            dis_real, _ = self.discriminator(dis_input_real)                   
            dis_fake, _ = self.discriminator(dis_input_fake)                   

            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
            gen_input_fake = outputs_img
            gen_fake, _ = self.discriminator(gen_input_fake)
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_loss += gen_gan_loss


        # generator l1 loss
            gen_l1_loss = self.l1_loss(outputs_img, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        #gen_l1_loss = self.l1_loss(outputs_img, images) * self.config.L1_LOSS_WEIGHT
            gen_loss += gen_l1_loss


        # generator perceptual loss
            gen_content_loss = self.perceptual_loss(outputs_img, images)
            gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
            gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs_img * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss
        #############################

        # create logs
        logs = [
            ("gLoss",gen_loss.item()),
            ("dLoss",dis_loss.item())
        ]

        return outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss

    # def forward(self, images, landmarks, masks):
    def forward(self, images, masks):
        images_masked = (images * (1 - masks).float()) + masks
        inputs = images_masked
        scaled_masks_tiny = F.interpolate(masks, size=[int(masks.shape[2] / 8), int(masks.shape[3] / 8)],
                                     mode='nearest')        
        
        scaled_masks_quarter = F.interpolate(masks, size=[int(masks.shape[2] / 4), int(masks.shape[3] / 4)],
                                     mode='nearest')
        scaled_masks_half = F.interpolate(masks, size=[int(masks.shape[2] / 2), int(masks.shape[3] / 2)],
                                     mode='nearest')
                                     
                                     
        pos1 = PositionalEncoding(48, 20000000)
        pos2 = PositionalEncoding(96, 8000000)
        pos3 = PositionalEncoding(192, 5000000)
        pos4 = PositionalEncoding(384, 2500000)
        pos1_dec = PositionalEncoding(96, 20000000)                             
        
        

        outputs_img = self.generator(inputs,masks,scaled_masks_half,scaled_masks_quarter,scaled_masks_tiny, pos1, pos2, pos3, pos4, pos1_dec)             

        return outputs_img

    def backward(self, gen_loss = None, dis_loss = None):

        self.scaler.scale(dis_loss).backward(retain_graph= True)
        
        self.scaler.scale(gen_loss).backward()
        
        self.scaler.step(self.dis_optimizer)
        
        self.scaler.step(self.gen_optimizer)
        self.scaler.update()
        
        # print(self.gen_scheduler.get_lr())

    def backward_joint(self, gen_loss = None, dis_loss = None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()



def abs_smooth(x):
    absx = torch.abs(x)
    minx = torch.min(absx,other=torch.ones(absx.shape).cuda())
    r = 0.5 *((absx-1)*minx + absx)
    return r
    
def PositionalEncoding(d_model, max_len=5000):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe.cuda()
