FROM gitpod/workspace-full-vnc

RUN sudo apt update
RUN sudo apt install xvfb
RUn sudo apt-get install python-opengl