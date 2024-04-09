import random
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image, ImageDraw, ImageFont
import time
import logging
import os

import torchvision
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid

from model import *
from mnist_data import *
from cifar_data import *
from MVTecAD_data import *


class image_data_set(Dataset):
    def __init__(self, data):
        self.images = data[:, :, :, None]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize(64, interpolation=InterpolationMode.BICUBIC),
            # 为mnist数据集进行超采样提高精度
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx])


def main(args):
    device = torch.device("cuda")

    # 初始化参数
    batch_size = 2048
    dg_train = 1
    GLR = 0.002
    DLR = 0.004

    # 加载训练数据
    # train_set = image_data_set(train)
    # train_set = image_data_set(x_train_resized)
    train_set = image_data_set(MVTecAD_grid_train_resized)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # 加载模型
    G = Generator().to(device)
    D = Discriminator().to(device)

    # 训练模式
    G.train()
    D.train()

    # 设置优化器
    optimizerG = torch.optim.Adam(G.parameters(), lr=GLR, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(D.parameters(), lr=DLR, betas=(0.0, 0.9))

    # 定义损失函数
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # 数据记录
    pic_g_loss = []
    pic_g_loss_nio = []
    pic_d_loss = []
    g_epoch = []
    g_epoch_nio = []
    d_epoch = []

    Time_Stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    fileroot = f'logs/{Time_Stamp}_bs-{batch_size}_epoch-{args.epochs}_dgt-{dg_train}_lr-g{GLR}-d{DLR}'
    os.makedirs(fileroot, exist_ok=True)
    os.makedirs(fileroot + "/output", exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler_console = logging.StreamHandler()
    handler_console.setLevel(logging.INFO)
    handler_file = logging.FileHandler(f'{fileroot}/log_file.log')
    handler_file.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_console.setFormatter(formatter)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_console)
    logger.addHandler(handler_file)

    """
    训练
    """

    # 开始训练
    for epoch in range(args.epochs * dg_train):
        # 定义初始损失
        log_g_loss, log_d_loss = 0.0, 0.0
        for images in train_loader:
            images = images.to(device)
            if (epoch + 1) % dg_train == 0:
                ## 训练判别器 Discriminator
                # 定义软标签 真标签和假标签   维度：（batch_size）
                label_real_d = torch.full((images.size(0),), random.uniform(0.9, 1)).to(device)
                label_fake_d = torch.full((images.size(0),), random.uniform(0, 0.1)).to(device)

                # 定义潜在变量z    维度：(batch_size,20,1,1)
                z = torch.randn(images.size(0), 20).to(device).view(images.size(0), 20, 1, 1).to(device)
                fake_images = G(z)

                # 真图像和假图像送入判别网络，得到 d_out_real、d_out_fake
                d_out_real, _ = D(images)
                d_out_fake, _ = D(fake_images)

                # 损失计算
                d_loss_real = criterion(d_out_real.view(-1), label_real_d)
                d_loss_fake = criterion(d_out_fake.view(-1), label_fake_d)
                d_loss = d_loss_real + d_loss_fake

                # 误差反向传播，更新损失
                optimizerD.zero_grad()
                d_loss.backward()
                optimizerD.step()
                log_d_loss += d_loss.item()

                pic_d_loss.append(log_d_loss)
                d_epoch.append(epoch / dg_train)

            # 生成器使用硬标签
            label_real_g = torch.full((images.size(0),), 1.0).to(device)
            label_fake_g = torch.full((images.size(0),), 0.0).to(device)

            ## 训练生成器 Generator
            # 定义潜在变量z    维度：(batch_size,20,1,1)
            z = torch.randn(images.size(0), 20).to(device).view(images.size(0), 20, 1, 1).to(device)
            fake_images = G(z)

            # 假图像喂入判别器，得到d_out_fake   维度：（batch_size,1,1,1）
            d_out_fake, _ = D(fake_images)

            # 损失计算
            g_loss = criterion(d_out_fake.view(-1), label_real_g)

            # 误差反向传播，更新损失
            optimizerG.zero_grad()
            g_loss.backward()
            # optimizerG.step()
            # 优化器按需迭代
            # if g_loss.item() / 128 > 0.02:
            optimizerG.step()

            log_g_loss += g_loss.item()

        pic_g_loss.append(log_g_loss)
        g_epoch.append(epoch)

        if (epoch + 1) % dg_train == 0:
            g_loss_nio = 0
            for i in range(epoch - dg_train + 1, epoch + 1):
                g_loss_nio += pic_g_loss[i]
            pic_g_loss_nio.append(g_loss_nio / dg_train)
            g_epoch_nio.append(epoch / dg_train)

            ## 累计一个epoch的损失，判别器损失和生成器损失分别存放到log_d_loss、log_g_loss中

        if (epoch + 1) % dg_train == 0:
            logger.info(
                f'epoch {epoch}, G_Loss:{log_g_loss / 128:.4f}, D_Loss:{log_d_loss / 128:.4f}, G_loss_avg:{pic_g_loss_nio[-1] / 128:.4f}')
        else:
            ## 打印损失
            logger.info(f'epoch {epoch}, G_Loss:{log_g_loss / 128:.4f}')

    z = torch.randn(256, 20).to(device).view(256, 20, 1, 1).to(device)
    fake_images = G(z)
    torchvision.utils.save_image(fake_images, f"{fileroot}/trainGenerate.jpg", nrow=16)

    # 绘图
    plt.Figure()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(g_epoch, pic_g_loss, linewidth=1, linestyle="solid", label="g_loss")
    plt.legend()
    plt.title("g_loss")
    plt.savefig(f"{fileroot}/g_loss.jpg", dpi=1000)
    plt.show()

    plt.Figure()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(d_epoch, pic_d_loss, linewidth=1, linestyle="solid", label="d_loss")
    plt.legend()
    plt.title("d_loss")
    plt.savefig(f"{fileroot}/d_loss.jpg", dpi=1000)
    plt.show()

    plt.Figure()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(g_epoch_nio, pic_g_loss_nio, linewidth=1, linestyle="solid", label="g_loss_avg")
    plt.legend()
    plt.title("g_loss_avg")
    plt.savefig(f"{fileroot}/g_loss_avg.jpg", dpi=1000)
    plt.show()

    """
    测试
    """

    ## 定义缺陷计算的得分
    def anomaly_score(input_image, fake_image, D):
        # Residual loss 计算
        residual_loss = torch.sum(torch.abs(input_image - fake_image), (1, 2, 3))

        # Discrimination loss 计算
        _, real_feature = D(input_image)
        _, fake_feature = D(fake_image)
        discrimination_loss = torch.sum(torch.abs(real_feature - fake_feature), (1))

        # 结合Residual loss和Discrimination loss计算每张图像的损失
        total_loss_by_image = 0.9 * residual_loss + 0.1 * discrimination_loss
        # 计算总损失，即将一个batch的损失相加
        total_loss = total_loss_by_image.sum()

        return total_loss, total_loss_by_image, residual_loss

    # 加载测试数据
    # test_set = image_data_set(test)
    # test_set = image_data_set(x_test_resized)
    test_set = image_data_set(MVTecAD_grid_test_resized)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    input_images = next(iter(test_loader)).to(device)

    # 定义潜在变量z  维度：（5，20，1，1）
    z = torch.randn(256, 20).to(device).view(256, 20, 1, 1)
    # z的requires_grad参数设置成Ture,让z可以更新
    z.requires_grad = True

    # 定义优化器
    z_optimizer = torch.optim.Adam([z], lr=1e-3)

    # 搜索z
    for epoch in range(5000):
        fake_images = G(z)
        loss, _, _ = anomaly_score(input_images, fake_images, D)

        z_optimizer.zero_grad()
        loss.backward()
        z_optimizer.step()

        if epoch % 1000 == 0:
            logger.info(f'epoch: {epoch}, loss: {loss:.0f}')

    fake_images = G(z)

    _, total_loss_by_image, _ = anomaly_score(input_images, fake_images, D)

    predictLabelPre = total_loss_by_image.cpu().detach().numpy()
    # predictLabel = (predictLabelPre[0] * 20 + predictLabelPre[1] + predictLabelPre[2] + predictLabelPre[3] +
    #                 predictLabelPre[4]) / 24

    logger.info(total_loss_by_image.cpu().detach().numpy())

    torchvision.utils.save_image(input_images, f"{fileroot}/testInput.jpg", nrow=16)
    torchvision.utils.save_image(fake_images, f"{fileroot}/testGenerate.jpg", nrow=16)

    def save_image_with_loss(img_tensor, loss, save_path):
        """将损失值绘制在图片上并保存"""
        img_pil = transforms.ToPILImage()(img_tensor.cpu())
        draw = ImageDraw.Draw(img_pil)
        # 尝试加载字体，如果失败则使用默认字体
        try:
            font = ImageFont.truetype("arial.ttf", size=20)  # 字体大小根据需要调整
        except IOError:
            font = ImageFont.load_default()
        text = f"{loss:.1f}"
        draw.text((10, 10), text, font=font, fill="white")
        img_pil.save(save_path)

    def save_batch_images(batch_images, save_path, nrow=16, padding=2):

        # 使用make_grid生成大图
        grid_img = make_grid(batch_images, nrow=nrow, padding=padding, normalize=True)
        # 转换为PIL Image以便保存
        ndarr = grid_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save(save_path)

    all_loss = []
    img_count = []
    for batch_idx, input_images in enumerate(test_loader):
        input_images = input_images.to(device)
        z = torch.randn(input_images.size(0), 20, 1, 1, device=device)
        fake_images = G(z)
        _, loss_values, _ = anomaly_score(input_images, fake_images, D)  # 假设已定义
        now_loss = 0
        for idx, (input_img, loss) in enumerate(zip(input_images, loss_values)):
            #now_loss += loss_values[idx].item()
            if loss_values[idx].item() > now_loss: now_loss = loss_values[idx].item()
        # now_loss = now_loss / input_images.size(0)
        all_loss.append(now_loss)
        img_count.append(batch_idx)
        logger.info(f"image {batch_idx} loss: {now_loss}")

        os.makedirs(fileroot + f"/output/test_img_{batch_idx}_{now_loss}", exist_ok=True)
        save_path = fileroot + f"/output/test_img_{batch_idx}_{now_loss}/total_image.png"
        save_batch_images(input_images, save_path, nrow=16)

        for idx, (input_img, loss) in enumerate(zip(input_images, loss_values)):
            loss = loss_values[idx].item()

            save_path = fileroot + f"/output/test_img_{batch_idx}_{now_loss}/{idx}_loss-{loss:.4f}.png"
            save_image_with_loss(input_img, loss, save_path)

    plt.Figure()
    plt.xlabel("img count")
    plt.ylabel("image loss")
    plt.plot(img_count, all_loss, "o", label="image loss")
    plt.legend()
    plt.title("image_loss")
    plt.savefig(f"{fileroot}/image_D.jpg", dpi=1000)
    plt.show()

    # # 切换模型为评估模式
    # D.eval()
    # G.eval()
    #
    # predictions = []
    # TP_count = 0
    # FN_count = 0
    # FP_count = 0
    # TN_count = 0
    #
    # # 遍历测试集
    # logger.info(f'get_d: {log_d_loss / 128}')
    # # logger.info(f'predictLabel: {predictLabel}')
    #
    # with torch.no_grad():
    #     for images in test_loader:
    #         images = images.to(device)
    #         fake_images = G(z)
    #
    #         if images.size(0) != fake_images.size(0):
    #             # 调整fake_image的大小以匹配input_image
    #             fake_images = fake_images[:images.size(0), ...]
    #
    #         # # 使用判别器预测图像
    #         # d_out, _ = D(images)
    #         #
    #         # # 将预测结果转换为二进制标签
    #         # binary_predictions = (d_out > log_d_loss / 128).view(-1).cpu().numpy().tolist()
    #
    #         _, d_out, _ = anomaly_score(images, fake_images, D)
    #
    #         # logger.info(d_out)
    #         # binary_predictions = (d_out > predictLabel).view(-1).cpu().numpy().tolist()
    #
    #         # 将预测结果添加到存储预测的列表中
    #         # predictions.extend(binary_predictions)
    #
    # Mid_Index = len(predictions) // 2
    # for index, predict in enumerate(predictions):
    #     # if predict:
    #     #     if index < Mid_Index:
    #     #         FP_count += 1
    #     #     else:
    #     #         TP_count += 1
    #     # else:
    #     #     if index < Mid_Index:
    #     #         TN_count += 1
    #     #     else:
    #     #         FN_count += 1
    #     if predict:
    #         if index < 21:
    #             TP_count += 1
    #         else:
    #             FP_count += 1
    #     else:
    #         if index < 21:
    #             FN_count += 1
    #         else:
    #             TN_count += 1
    #
    # logger.info(f'FP: {FP_count}, FN: {FN_count}, TN: {TN_count}, TP: {TP_count}')
    #
    # Conf_Matrix = [[TP_count, FN_count],
    #                [FP_count, TN_count]]
    # Precision = TP_count / (TP_count + FP_count)
    # Recall = TP_count / (TP_count + FN_count)
    # MAA = TN_count / (TN_count + FP_count)
    # F1_Measure = 2 * Precision * Recall / (Precision + Recall)
    #
    # logger.info('Conf_Matrix:')
    # logger.info(Conf_Matrix)
    # logger.info(f'Precision: {Precision}, Recall: {Recall}, Majority Accuracy: {MAA}, F1_Measure: {F1_Measure}')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch_size", default=2048, type=int)
    parser.add_argument("--epochs", default=100, type=int, metavar="N",
                        help="number of total epochs to train")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
