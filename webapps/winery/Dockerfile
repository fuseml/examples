FROM python:3.7-slim-stretch

RUN pip3 install --no-cache-dir streamlit numpy pandas

# set streamlit config for running in docker
RUN mkdir -p /root/.streamlit && \
bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml' && \
bash -c 'echo -e "[server]\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > /root/.streamlit/config.toml'

COPY wineapp.py webapp/app.py

CMD streamlit run webapp/app.py

EXPOSE 8501
