CREATE TABLE Base (id INT AUTO_INCREMENT primary key, entity varchar(256), relation varchar(256), value varchar(256), freebase_id varchar(256), freebase_entity varchar(256), link_scroe varchar(256), link_am_score varchar(256));

create TABLE Freebase (id INT AUTO_INCREMENT primary key, freebase_id varchar(256), freebase_entity varchar(256));

create table EntitySelect (id INT AUTO_INCREMENT primary key, base_id varchar(256), entity varchar(256), relation varchar(256), value varchar(256), freebase_id varchar(256), freebase_entity varchar(256), link_scroe varchar(256), link_am_score varchar(256));

create table NoiseEntity (id INT AUTO_INCREMENT primary key, base_id varchar(256), entity varchar(256), relation varchar(256), value varchar(256), freebase_id varchar(256), freebase_entity varchar(256), link_scroe varchar(256), link_am_score varchar(256));



select b.id, entity, relation, value, w.id, freebase_entity from BaseAll b inner join WordSelect w on b.freebase_id = w.freebase_id;



select f.freebase_id, f.freebase_entity, b.id, b.entity, b.relation, b.value, b.link_scroe, b.link_am_score from Freebase f 
inner join Base b  
on b.freebase_id = f.freebase_id
where f.id <= 150;


create view temp as
select freebase_id from Base where link_am_score != "nan" and link_am_score > 0.2 group by freebase_id having count(freebase_id) > 6 limit 150;



insert into NoiseEntity (base_id, entity, relation, value, freebase_id, freebase_entity, link_scroe, link_am_score) 
select b.id, b.entity, b.relation, b.value, b.freebase_id, b.freebase_entity, b.link_scroe, b.link_am_score from Base b 
inner join temp t 
on t.freebase_id = b.freebase_id
where b.link_am_score != "nan"