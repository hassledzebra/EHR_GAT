import h5py
import torch
import torch
import torch.nn.functional as F
from pyspark.sql.functions import *
from torch_geometric.nn import GATConv
from torch_geometric.data import HeteroData
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType


def graph_prep(cohort, demo, como, topn):
    print('-------preprocess data--------')
    cohort_new = cohort
    cohort_como = como
    cohort_demo = demo
    cohort_como_reformat= cohort_como.withColumn('comorbidityid',expr("regexp_replace(comorbidityid, '([0-9a-zA-Z]+\\.[0-9a-zA-Z]).*', '$1')").cast("string"))
    top_comorbidity = get_top_comorbidity(cohort_como_reformat, topn=topn)
    cohort_como_length = cohort_como_reformat.join(cohort_new.select('personid','date'),'personid').withColumn('len_como',months_between(col('date'),col('effectivedate')))
    cohort_como_length_filter = cohort_como_length.filter(col('comorbidityid').isin(top_comorbidity))
    cohort_demo_processed = cohort_demo.withColumn('age_at_diagnosis',months_between(col('date'),col('birthdate.value'))/12)
    cohort_demo_processed = cohort_demo_processed.drop('birthdate')
    
    # Preprocess gender column
    df = cohort_demo_processed
    df = df.withColumn("gender",
                       when(lower(col("gender")).isin("male"), 1)
                       .when(lower(col("gender")).isin("female"), 2)
                       .when(col("gender").isNull(), 0)
                       .otherwise(3))

    # Preprocess race column
    df = df.withColumn("race",
                       when(lower(col("race")).like("%white%") | lower(col("race")).like("%caucasian%"), 1)
                       .when(lower(col("race")).like("%black%") | lower(col("race")).like("%african%"), 2)
                       .when(lower(col("race")).like("%hispanic%"), 3)
                       .when(lower(col("race")).like("%asian%") | lower(col("race")).like("%chinese%") | lower(col("race")).like("%korean%") | lower(col("race")).like("%japanese%"), 4)
                       .when(lower(col("race")).like("%indian american%") | lower(col("race")).like("%native american%"), 5)
                       .when(col("race").isNull(), 0)
                       .otherwise(6))
#     df.show()
    cohort_demo_processed = df
    print('completed')
    
    print('-------create graph--------')
    
    person_vertice = cohort_demo_processed
    person_vertice = person_vertice.withColumn("id", col('personid')).drop('personid','date')
    diagnosis_vertice = cohort_como_length_filter.select("comorbidityid").distinct()
    diagnosis_vertice = diagnosis_vertice.withColumn("id", col('comorbidityid')).drop(col('comorbidityid'))
    
    personid_diagnosis_edges = cohort_new.join(cohort_como_length_filter,'personid').select("personid", "comorbidityid",'len_como')
    # personid_diagnosis_edges = personid_diagnosis_edges.withColumn("cormorbidity_year", col('comorbidity_year'))
    personid_diagnosis_edges = personid_diagnosis_edges.withColumn("src", col('personid'))
    personid_diagnosis_edges = personid_diagnosis_edges.withColumn("dst", col('comorbidityid'))

    diagnosis_personid_edges = personid_diagnosis_edges.withColumn("src", col('comorbidityid'))
    diagnosis_personid_edges = personid_diagnosis_edges.withColumn("dst", col('personid'))

    personid_diagnosis_edges = personid_diagnosis_edges.drop(col('personid')).drop(col('comorbidityid'))
    diagnosis_personid_edges = diagnosis_personid_edges.drop(col('personid')).drop(col('comorbidityid'))
    all_edges = personid_diagnosis_edges
    # revised to put comorbidity indexes at the beginning


    # Mapping person and comorbidity IDs to indices
    comorbidity_id_to_idx = {id_: idx  for idx, id_ in enumerate(diagnosis_vertice.toPandas()['id'])}
    person_id_to_idx = {id_: idx + len(comorbidity_id_to_idx) for idx, id_ in enumerate(person_vertice.toPandas()['id'])}

    id_to_idx = {**comorbidity_id_to_idx, **person_id_to_idx}
    # prompt: convert id of person_vertice using id_to_idx

    # Assuming 'person_vertice' is a PySpark DataFrame


    # Create a UDF to map person IDs to indices
    map_person_id_udf = udf(lambda id_: id_to_idx.get(id_), IntegerType())

    # Apply the UDF to the 'id' column of 'person_vertice'
    person_vertice_indexed = person_vertice.withColumn("id_index", map_person_id_udf(col("id")))

    # Show the updated DataFrame
#     person_vertice_indexed.show()
    # prompt: convert id of diagnosis_vertice using id_to_idx

    # Create a UDF to map comorbidity IDs to indices
    map_comorbidity_id_udf = udf(lambda id_: id_to_idx.get(id_), IntegerType())

    # Apply the UDF to the 'id' column of 'diagnosis_vertice'
    diagnosis_vertice_indexed = diagnosis_vertice.withColumn("id_index", map_comorbidity_id_udf(col("id")))

    # Show the updated DataFrame
#     diagnosis_vertice_indexed.show()
    # prompt: convert src and dst of all_edges using id_to_idx

    # Create a UDF to map IDs to indices for 'src' column
    map_src_id_udf = udf(lambda id_: id_to_idx.get(id_), IntegerType())

    # Apply the UDF to the 'src' column of 'all_edges'
    all_edges_indexed = all_edges.withColumn("src_index", map_src_id_udf(col("src")))

    # Create a UDF to map IDs to indices for 'dst' column
    map_dst_id_udf = udf(lambda id_: id_to_idx.get(id_), IntegerType())

    # Apply the UDF to the 'dst' column of 'all_edges_indexed'
    all_edges_indexed = all_edges_indexed.withColumn("dst_index", map_dst_id_udf(col("dst")))

    # Show the updated DataFrame
#     all_edges_indexed.show()
    # prompt: format person_vertice to x_person, format diagnosis_vertice to x_diagnosis

    # Assuming 'person_vertice' is a PySpark DataFrame with columns: 'id_index', 'age_at_diagnosis', 'gender', 'race'
    x_person = torch.tensor(person_vertice_indexed.select("id_index", "age_at_diagnosis", "gender", "race")
                             .toPandas().values, dtype=torch.float)

    # Assuming 'diagnosis_vertice' is a PySpark DataFrame with column: 'id_index'
    x_diagnosis = torch.tensor(diagnosis_vertice_indexed.select("id_index")
                                .toPandas().values, dtype=torch.int)


    # Handle null values in x_person
    x_person = torch.nan_to_num(x_person, nan=0.0)  # Replace NaN with 0.0

#     print(x_person)
    # prompt: pad x_diagnosis with 0 so that it has the same dimensions except for dimension 0

    # Pad x_diagnosis with zeros
    padding_size = x_person.shape[1] - x_diagnosis.shape[1]
    padding = torch.zeros(x_diagnosis.shape[0], padding_size)
    x_diagnosis_padded = torch.cat([x_diagnosis, padding], dim=1).float()


    # Merge person and diagnosis features, put diagnosis first
    x_combined = torch.cat([x_diagnosis_padded, x_person], dim=0)


    # Extract source and destination indices and edge attributes
    src_indices = all_edges_indexed.select("src_index").rdd.flatMap(lambda x: x).collect()
    dst_indices = all_edges_indexed.select("dst_index").rdd.flatMap(lambda x: x).collect()
    # edge_attrs = all_edges_indexed.select("len_como").toPandas().values

    edge_index_person_to_diagnosis = torch.tensor([src_indices, dst_indices]).contiguous()
#     print(edge_index_person_to_diagnosis)
    edge_index_diagnosis_to_person = torch.tensor([dst_indices, src_indices]).contiguous()
    # print(edge_index_person_to_diagnosis)
    edge_attrs = all_edges_indexed.select("len_como").rdd.map(lambda x: x[0]).collect()
    # edge_attrs = all_edges_indexed.select("len_como").toPandas().values

    # Create edge attribute tensor
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)  # Unsqueeze to match the expected shape
    
    # Create a UDF to map person IDs to indices for labels
    map_person_id_for_label_udf = udf(lambda id_: id_to_idx.get(id_), IntegerType())

    # Apply the UDF to the 'personid' column of 'cohort_new' and select 'EPI' as label
    cohort_new_indexed = cohort_new.withColumn("id_index", map_person_id_for_label_udf(col("personid"))).select('id_index','EPI')

    # align with x_person by doing sorting
    cohort_new_indexed = cohort_new_indexed.orderBy('id_index')
    # display(cohort_new_indexed.show(5))

    # Extract labels and convert to tensor
    y = torch.tensor(cohort_new_indexed.select("EPI").toPandas().values.astype(int), dtype=torch.long).squeeze()
    

    # prompt: transform the cohort_new so that personid is converted to indexes using id_to_idx. EPI is the label. Make sure the label is aligned with id in x_person. Load the label to y and prepare data to be loaded using dataloader

    # Create a UDF to map person IDs to indices for labels
    map_person_id_for_label_udf = udf(lambda id_: id_to_idx.get(id_), IntegerType())

    # Apply the UDF to the 'personid' column of 'cohort_new' and select 'EPI' as label
    cohort_new_indexed = cohort_new.withColumn("id_index", map_person_id_for_label_udf(col("personid"))).select('id_index','EPI')

    # align with x_person by doing sorting
    cohort_new_indexed = cohort_new_indexed.orderBy('id_index')
    # display(cohort_new_indexed.show(5))

    # Extract labels and convert to tensor
    y = torch.tensor(cohort_new_indexed.select("EPI").toPandas().values.astype(int), dtype=torch.long).squeeze()
    
    # Create HeteroData object
    data1 = HeteroData()
    data1.x = x_combined
    data1.edge_index = edge_index_person_to_diagnosis
    data1.edge_attr = edge_attr
    data1.y = y
    
    print('structure of created graph:')
    print(data1)
    

    print('completed')
    print('-------write data to h5 file--------')
    # Create an h5 file
    with h5py.File('heterodata_diag'+str(topn)+'_first.h5', 'w') as f:
      # Store node features
      f.create_dataset('x', data=data1.x.numpy())

      # Store edge index
      f.create_dataset('edge_index', data=data1.edge_index.numpy())

      # Store edge attributes
      f.create_dataset('edge_attr', data=data1.edge_attr.numpy())

      # Store labels
      f.create_dataset('y', data=data1.y.numpy())

    
    
def get_top_comorbidity(como, topn):
    top_como = como.groupBy('comorbidityid').count().orderBy(col('count').desc())\
        .select(collect_list('comorbidityid')).collect()[0]\
            .__getitem__('collect_list(comorbidityid)')[0:topn]

#     top_como_coded = ["n"+x.replace('.','_') for x in top_como]
#     out = {top_como[i]:top_como_coded[i] for i in range(len(top_como))} # store in a dictionary
    print('selected comorbidities:',top_como)
    return top_como