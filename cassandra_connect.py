from cassandra.cluster import Cluster

clstr = Cluster()
session = clstr.connect()
session.execute("create keyspace if not exists mykeyspace with replication={'class':'SimpleStrategy', 'replication_factor':3 };")

session = clstr.connect('mykeyspace')
qry = """
create table if not exists traffic_fault (
    id int,
    location varchar,
    vehicle varchar,
    rule_broken varchar,
    license_plate varchar,
    PRIMARY KEY (id)
);"""

session.execute(qry)

#session.execute("insert into traffic_fault (id, location, vehicle, rule_broken, license_plate) values (2, 'shivaginagar', 'car','broke signal','KA7338');")

#rows = session.execute("select * from traffic_fault;")

#for row in rows:
 #   print(row)
