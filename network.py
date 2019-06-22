import torch
import torch.nn as nn
import torch.nn.functional as F

def pad2d(x, n):
    pad=nn.ConstantPad2d((n-n//2,n//2,n-n//2,n//2), 0)
    return pad(x)

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, p):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=p, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=p, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x
        
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
        
class U_Net(nn.Module):
    def __init__(self,img_ch=4,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64, p=0)
        self.Conv2 = conv_block(ch_in=64,ch_out=128, p=0)
        self.Conv3 = conv_block(ch_in=128,ch_out=256, p=0)
        self.Conv4 = conv_block(ch_in=256,ch_out=512, p=0)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024, p=1)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512, p=1)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256, p=1)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128, p=1)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64, p=1)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        #print('x: ', x.size())
        x1 = self.Conv1(x)
        #print('x1: ', x1.size())

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        #print('x2: ', x2.size())
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        #print('x3: ', x3.size())

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        #print('x4: ', x4.size())

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        #print('x5: ', x5.size())

        # decoding + concat path
        d5 = self.Up5(x5)
        #print('d5: ', d5.size())
        d5 = torch.cat((x4,d5),dim=1)
        #print('d5 after cat: ', d5.size())
        d5 = self.Up_conv5(d5)
        #print('d5 after upcomv: ', d5.size())
        
        d4 = self.Up4(d5)
        #print('d4: ', d4.size())

        d4 = pad2d(d4, x3.shape[2]-d4.shape[2])
        #print('d4 after padding: ', d4.size())
        d4 = torch.cat((x3,d4),dim=1)
        #print('d4 after cat: ', d4.size())
        d4 = self.Up_conv4(d4)
        #print('d4 after upconv:', d4.size())

        d3 = self.Up3(d4)
        #print('d3: ', d3.size())
        d3 = pad2d(d3, x2.shape[2]-d3.shape[2])
        d3 = torch.cat((x2,d3),dim=1)
        #print('d3 after cat: ', d3.size())
        d3 = self.Up_conv3(d3)
        #print('d3 after upconv: ', d3.size())

        d2 = self.Up2(d3)
        #print('d2: ', d2.size())
        d2 = pad2d(d2, x1.shape[2]-d2.shape[2])
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        #print('d2: ', d2.size())
        d2 = pad2d(d2, x.shape[2]-d2.shape[2])

        d1 = self.Conv_1x1(d2)
        #print('d1: ', d1.size())
        

        return d1
