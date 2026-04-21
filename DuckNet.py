import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class WidescopeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.widescope = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.widescope(x)

class MidscopeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.midscope = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.midscope(x)

class SeparatedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.separate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,7), padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(7,1), padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.separate(x)

class ResnetConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )        
        
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.final_bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x1 = self.skip(x)
        x = self.double_conv(x)
        x_final = x + x1
        return self.final_bn(x_final)

class DuckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn_first = nn.BatchNorm2d(in_channels)
        self.bn_last = nn.BatchNorm2d(out_channels)

        self.branch1 = WidescopeConv(in_channels, out_channels)
        self.branch2 = MidscopeConv(in_channels, out_channels)
        self.branch3 = SeparatedConv(in_channels, out_channels)

        self.branch4 = nn.Sequential(
            ResnetConv(in_channels, out_channels),
            ResnetConv(out_channels, out_channels),
            ResnetConv(out_channels, out_channels),
        )

        self.branch5 = nn.Sequential(
            ResnetConv(in_channels, out_channels),
            ResnetConv(out_channels, out_channels),
        )

        self.branch6 = ResnetConv(in_channels, out_channels)
        
    def forward(self, x):
        x = self.bn_first(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x6 = self.branch6(x)

        sum_branches = x1 + x2 + x3 + x4 + x5 + x6

        final = self.bn_last(sum_branches)
        return final

class DuckNet(nn.Module):
    def __init__(self, in_channels, num_classes=1, starting_filters = 17):
        super().__init__()

        filters = [starting_filters * 2**(i+1) for i in range(5)]

        self.p1 = nn.Conv2d(in_channels, filters[0], kernel_size=2, stride =2, padding=0)
        self.p2 = nn.Conv2d(filters[0], filters[1], kernel_size=2, stride =2, padding=0)
        self.p3 = nn.Conv2d(filters[1], filters[2], kernel_size=2, stride =2, padding=0)
        self.p4 = nn.Conv2d(filters[2], filters[3], kernel_size=2, stride =2, padding=0)
        self.p5 = nn.Conv2d(filters[3], filters[4], kernel_size=2, stride =2, padding=0)

        self.t0 = DuckBlock(in_channels, starting_filters)
        
        self.l1i = nn.Conv2d(starting_filters, filters[0], kernel_size=2, stride=2, padding=0)
        self.t1 = DuckBlock(filters[0], filters[0])

        self.l2i = nn.Conv2d(filters[0], filters[1], kernel_size=2, stride=2, padding=0)
        self.t2 = DuckBlock(filters[1], filters[1])


        self.l3i = nn.Conv2d(filters[1], filters[2], kernel_size=2, stride=2, padding=0)
        self.t3 = DuckBlock(filters[2], filters[2])


        self.l4i = nn.Conv2d(filters[2], filters[3], kernel_size=2, stride=2, padding=0)
        self.t4 = DuckBlock(filters[3], filters[3])

        self.l5i = nn.Conv2d(filters[3], filters[4], kernel_size=2, stride=2, padding=0)

        self.t51 = ResnetConv(filters[4], filters[4])
        self.t53 = ResnetConv(filters[4], filters[3])

        self.q4 = DuckBlock(filters[3], filters[2])
        self.q3 = DuckBlock(filters[2], filters[1])
        self.q6 = DuckBlock(filters[1], filters[0])
        self.q1 = DuckBlock(filters[0], starting_filters)
        self.z1 = DuckBlock(starting_filters, starting_filters)

        self.output = nn.Conv2d(starting_filters, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)

        t0 = self.t0(x)
        l1i = self.l1i(t0)
        s1 = l1i + p1

        t1 = self.t1(s1)
        l2i = self.l2i(t1)
        s2 = l2i + p2

        t2 = self.t2(s2)
        l3i = self.l3i(t2)
        s3 = l3i + p3

        t3 = self.t3(s3)
        l4i = self.l4i(t3)
        s4 = l4i + p4

        t4 = self.t4(s4)
        l5i = self.l5i(t4)
        s5 = l5i + p5

        t51 = self.t51(s5)
        t53 = self.t53(t51)

        l5o = F.interpolate(t53, scale_factor=2, mode='nearest')
        c4 = l5o + t4
        q4 = self.q4(c4)

        l4o = F.interpolate(q4, scale_factor=2, mode='nearest')
        c3 = l4o + t3
        q3 = self.q3(c3)

        l3o = F.interpolate(q3, scale_factor=2, mode='nearest')
        c2 = l3o + t2
        q6 = self.q6(c2)

        l2o = F.interpolate(q6, scale_factor=2, mode='nearest')
        c1 = l2o + t1
        q1 = self.q1(c1)

        l1o = F.interpolate(q1, scale_factor=2, mode='nearest')
        c0 = l1o + t0

        z1 = self.z1(c0)
        output = self.output(z1)

        return output

