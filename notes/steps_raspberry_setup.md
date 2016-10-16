1. Initially I downloaded and copied Raspbian Lite onto SD by mistake. Not what I needed. Went back and downloaded NOOBS full instead. After downloading, I unzipped the zip file, opened the resulting folder, copied the contents to the SD. Make sure the SD card is formatted as FAT32 (MS-DOS FAT).

2. Inserted microSD to RaspPi. Booted up. Got rainbow screen. Raspbian was the only OS to select, so I checked the box for it and installed. Took a few minutes. Had to reboot once because it got stuck after extracting 20 MB.

3. Connect to wi-fi. Find network icon in upper-right hand corner of desktop.

4. You can email python scripts to yourself. (Or bluetooth file transfer, if you ever figure out how the hell it works.) You emailed the stream_client.py file to yourself, and then downloaded it onto the RaspPi desktop.

5. A snag encountered when trying to connect the sockets. The RaspPi is the client; your laptop is the host. The IP address in both scripts should be the IP address of the host.

  - If RaspPi outputs an error when running the client_stream.py ("No route to host."), you should try running the SSH connection command from your laptop (host).
    From command line:
      >> ssh pi@192.168.1.78 [or whatever the pi's IP address is]
        [enter pi's password: raspberry]
      pi@raspberrypi:~ $ logout

      >> python collect_training_data_py.py

      [and then run the client_stream.py from the RaspPi]

      For more info: http://raspberrypi.stackexchange.com/questions/37149/raspberry-pi-connected-to-internet-but-cant-ssh-or-ping
      (see answer from Jamie Cox on Jun 24, 2016)

6. 
