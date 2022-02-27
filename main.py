import torch
from generator import Generator
from discriminator import PatchGAN


# Dummy Data
horse = torch.randn((1, 3, 26, 26))
zebra = torch.randn((1, 3, 26, 26))

# Training
in_channels = 3
epochs = 1
dis_epochs = 1

main_loss = torch.nn.MSELoss()
loss2 = torch.nn.L1Loss()

gan_horse = Generator(in_channels)
gan_zebra = Generator(in_channels)
dis_horse = PatchGAN(in_channels)
dis_zebra = PatchGAN(in_channels)

gan_opt = torch.optim.Adam(list(gan_zebra.parameters()) + list(gan_horse.parameters()), lr=0.001)
dis_opt = torch.optim.Adam(list(dis_zebra.parameters()) + list(dis_horse.parameters()), lr=0.001)

for epoch in range(epochs):
    fake_zebra = gan_zebra(horse)
    fake_horse = gan_horse(zebra)

    for _ in range(dis_epochs):
        dis_horse_real = dis_horse(horse)
        dis_horse_fake = dis_horse(fake_horse)
        dis_zebra_real = dis_zebra(zebra)
        dis_zebra_fake = dis_zebra(fake_zebra)

        dis_horse_real_loss = main_loss(dis_horse_real, torch.ones_like(dis_horse_fake))
        dis_horse_fake_loss = main_loss(dis_horse_fake, torch.zeros_like(dis_horse_fake))

        dis_zebra_fake_loss = main_loss(dis_zebra_fake, torch.zeros_like(dis_zebra_fake))
        dis_zebra_real_loss = main_loss(dis_zebra_real, torch.zeros_like(dis_zebra_real))

        dis_loss = (dis_zebra_fake_loss + dis_zebra_real_loss + dis_horse_fake_loss + dis_horse_real_loss)
        print(f"for epoch {epoch} disc loss is {dis_loss}")
        dis_loss.backward(retain_graph=True)
        dis_opt.step()
        dis_opt.zero_grad()

    fake_zebra_from_fake_horse = gan_zebra(fake_horse)
    fake_horse_from_fake_zebra = gan_horse(fake_zebra)

    zebra_loss_fake = loss2(fake_zebra_from_fake_horse, zebra)
    horse_loss_fake = loss2(fake_horse_from_fake_zebra, horse)

    fake_zebra_from_zebra = gan_zebra(zebra)
    fake_horse_from_horse = gan_horse(horse)

    horse_loss_real = loss2(fake_horse_from_horse, horse)
    zebra_loss_real = loss2(fake_zebra_from_zebra, zebra)

    horse_loss = horse_loss_real + horse_loss_fake
    zebra_loss = zebra_loss_real + zebra_loss_fake

    z_main_loss = main_loss(dis_zebra_fake, torch.ones_like(dis_zebra_fake))
    h_main_loss = main_loss(dis_horse_fake, torch.ones_like(dis_horse_fake))

    gan_loss = zebra_loss + horse_loss + z_main_loss + h_main_loss
    print(f'for epoch {epoch} gan loss is {gan_loss}')

    gan_loss.backward()

    gan_opt.step()
    gan_opt.zero_grad()
