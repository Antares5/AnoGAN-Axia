from torch import nn

"""定义调整后的生成器网络结构"""
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def CBA(in_channel, out_channel, kernel_size=4, stride=2, padding=1, activation=nn.ReLU(inplace=True), bn=True):
            seq = [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)]
            if bn:
                seq.append(nn.BatchNorm2d(out_channel))
            seq.append(activation)
            return nn.Sequential(*seq)

        seq = [CBA(20, 64*16, stride=1, padding=0)]  # 调整起始层
        seq += [CBA(64*16, 64*8)]
        seq += [CBA(64*8, 64*4)]
        seq += [CBA(64*4, 64*2)]
        seq += [CBA(64*2, 64)]
        seq += [CBA(64, 32)]  # 新增层以适应更大的尺寸
        seq += [CBA(32, 1, activation=nn.Tanh(), bn=False)]

        self.generator_network = nn.Sequential(*seq)

    def forward(self, z):
        return self.generator_network(z)

"""定义调整后的判别器网络结构"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def CBA(in_channel, out_channel, kernel_size=4, stride=2, padding=1, activation=nn.LeakyReLU(0.1, inplace=True)):
            seq = [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)]
            seq.append(nn.BatchNorm2d(out_channel))
            seq.append(activation)
            return nn.Sequential(*seq)

        seq = [CBA(1, 32)]  # 新增层以适应更大的尺寸
        seq += [CBA(32, 64)]
        seq += [CBA(64, 64*2)]
        seq += [CBA(64*2, 64*4)]
        seq += [CBA(64*4, 64*8)]
        seq += [CBA(64*8, 64*16)]  # 调整结束层
        self.feature_network = nn.Sequential(*seq)

        self.critic_network = nn.Conv2d(64*16, 1, kernel_size=4, stride=1)

    def forward(self, x):
        feature = self.feature_network(x)
        out = self.critic_network(feature)
        return out, feature.view(feature.size(0), -1)
