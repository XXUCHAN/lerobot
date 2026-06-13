FROM apache/spark:4.0.3-scala2.13-java17-python3-ubuntu

USER root

ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${PATH}"
ENV PYSPARK_PYTHON=python3

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

RUN if ! getent passwd 1001 >/dev/null; then \
      echo "spark:x:1001:0:Spark User:/opt/spark:/bin/bash" >> /etc/passwd; \
    fi

RUN ln -sf /usr/bin/python3 /usr/local/bin/python

USER 1001
