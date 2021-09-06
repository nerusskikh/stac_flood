wget -O flood-train-images.tgz "https://drivendata-prod.s3.amazonaws.com/data/81/public/flood-train-images.tgz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYVI2LMPSY%2F20210902%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210902T121144Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c16aa64219cb43a007ba4921c5c7b8da599e868e40c34d95670670b232d561d1"
wget -O flood-train-labels.tgz "https://drivendata-prod.s3.amazonaws.com/data/81/public/flood-train-labels.tgz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYVI2LMPSY%2F20210902%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210902T120706Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=8e2ee295206c3c5fb2cc599e733881fac694577526db6c7da9238dc0ec0efd94"
mkdir -p data
tar -xf flood-train-images.tgz -C data/
tar -xf flood-train-labels.tgz -C data/
rm flood-train-images.tgz
rm flood-train-labels.tgz